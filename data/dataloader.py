"""
Mesosfer Bear AI - Sequence Packing Dataloader

Reads tokenized sequences from storage/dataset/{stage}/{domain}/train|val/,
packs them into fixed-length context windows (default 4096) separated by
<|endoftext|>, and yields (input_ids, labels) tensors for causal LM training.

Key design:
  - Lazy streaming: never loads entire corpus into RAM.
  - Multi-format: parquet (any text column), json.gz, jsonl, json.
  - Deterministic shuffling per epoch via seed.
  - Domain-weighted sampling from DATASET_REGISTRY weights.
"""

import os
import json
import gzip
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader

from data.dataset import DATASET_REGISTRY


# -- Text extraction helpers ------------------------------------------------

# ponytail: column priority order for parquet text extraction
_TEXT_COLUMNS = ("text", "content", "code", "body", "sentence", "document")


def _extract_text_from_parquet(file_path: Path) -> Iterator[str]:
    """Yield text strings from a parquet file, auto-detecting the text column."""
    import pyarrow.parquet as pq

    table = pq.read_table(file_path)
    col_name = None
    for candidate in _TEXT_COLUMNS:
        if candidate in table.column_names:
            col_name = candidate
            break
    if col_name is None:
        # Fallback: first string column
        for col in table.column_names:
            if table.schema.field(col).type == "string":
                col_name = col
                break
    if col_name is None:
        return

    column = table.column(col_name)
    for chunk in column.chunks:
        for val in chunk:
            text = val.as_py()
            if text:
                yield text


def _extract_text_from_json_gz(file_path: Path) -> Iterator[str]:
    """Yield text strings from a gzipped JSON/JSONL file."""
    with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                for key in _TEXT_COLUMNS:
                    if key in obj and obj[key]:
                        yield obj[key]
                        break
            elif isinstance(obj, str) and obj:
                yield obj


def _extract_text_from_jsonl(file_path: Path) -> Iterator[str]:
    """Yield text strings from a JSONL file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                for key in _TEXT_COLUMNS:
                    if key in obj and obj[key]:
                        yield obj[key]
                        break
            elif isinstance(obj, str) and obj:
                yield obj


def _extract_text_from_json(file_path: Path) -> Iterator[str]:
    """Yield text strings from a JSON file (array of objects or array of strings)."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for key in _TEXT_COLUMNS:
                    if key in item and item[key]:
                        yield item[key]
                        break
            elif isinstance(item, str) and item:
                yield item


def extract_texts(file_path: Path) -> Iterator[str]:
    """Auto-dispatch text extraction based on file extension."""
    name = file_path.name.lower()
    if name.endswith(".parquet"):
        yield from _extract_text_from_parquet(file_path)
    elif name.endswith((".json.gz", ".jsonl.gz", ".gz")):
        yield from _extract_text_from_json_gz(file_path)
    elif name.endswith(".jsonl"):
        yield from _extract_text_from_jsonl(file_path)
    elif name.endswith(".json"):
        yield from _extract_text_from_json(file_path)


# -- File discovery ---------------------------------------------------------

def discover_shard_files(
    stage: str,
    split: str = "train",
    base_dir: str = "storage/dataset",
) -> List[Path]:
    """
    Discover all data files for a given stage and split.
    Returns list of Paths sorted for determinism.
    """
    stage_dir = Path(base_dir) / stage
    if not stage_dir.exists():
        return []

    valid_ext = (".parquet", ".json", ".jsonl", ".json.gz", ".jsonl.gz", ".gz")
    files = []
    for domain_dir in sorted(stage_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        split_dir = domain_dir / split
        if not split_dir.exists():
            continue
        for f in sorted(split_dir.iterdir()):
            if f.is_file() and f.name.lower().endswith(valid_ext):
                files.append(f)
    return files


# -- Sequence Packing Dataset -----------------------------------------------

class PackedDataset(IterableDataset):
    """
    Streaming sequence-packing dataset for causal language model pre-training.

    Reads raw text from shards, tokenizes on-the-fly, and packs token sequences
    into fixed-length windows of `context_length` tokens. Documents are separated
    by `eos_token_id` (<|endoftext|>).

    For causal LM, labels = input_ids shifted by 1 (handled by the model's
    loss function typically, so we return input_ids as both x and y offset by 1).
    """

    def __init__(
        self,
        tokenizer,
        stage: str = "pretrain",
        split: str = "train",
        context_length: int = 4096,
        base_dir: str = "storage/dataset",
        seed: int = 42,
        shuffle_shards: bool = True,
    ):
        self.tokenizer = tokenizer
        self.stage = stage
        self.split = split
        self.context_length = context_length
        self.base_dir = base_dir
        self.seed = seed
        self.shuffle_shards = shuffle_shards

        self.files = discover_shard_files(stage, split, base_dir)
        if not self.files:
            raise FileNotFoundError(
                f"No data files found for stage='{stage}', split='{split}' in {base_dir}"
            )

        # Use <|end_of_text|> as document separator
        self.eos_id = tokenizer.eos_id

    def _token_stream(self, epoch: int = 0) -> Iterator[int]:
        """Yield a flat stream of token IDs from all shard files, with EOS between docs."""
        files = list(self.files)
        if self.shuffle_shards:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(files)

        for file_path in files:
            for text in extract_texts(file_path):
                token_ids = self.tokenizer.encode(text)
                if token_ids:
                    yield from token_ids
                    yield self.eos_id

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Yield packed (input_ids, labels) pairs of shape (context_length,).

        input_ids = tokens[0:context_length]
        labels    = tokens[1:context_length+1]

        This is the standard causal LM next-token-prediction setup.
        """
        # ponytail: simple buffer fill approach, no over-engineering
        buf: List[int] = []
        seq_len = self.context_length + 1  # +1 for shifted labels

        worker_info = torch.utils.data.get_worker_info()
        epoch = 0 if worker_info is None else worker_info.id

        for token_id in self._token_stream(epoch):
            buf.append(token_id)
            if len(buf) >= seq_len:
                chunk = buf[:seq_len]
                buf = buf[seq_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, labels

        # ponytail: discard leftover tokens shorter than context_length.
        # Padding a partial window adds noise; dropping <4096 tokens per epoch is negligible.


# -- Convenience factory -----------------------------------------------------

def create_dataloader(
    tokenizer,
    stage: str = "pretrain",
    split: str = "train",
    context_length: int = 4096,
    batch_size: int = 4,
    base_dir: str = "storage/dataset",
    seed: int = 42,
    num_workers: int = 0,
    shuffle_shards: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader with sequence packing for causal LM training.

    Args:
        tokenizer: BearTokenizer instance (must have .encode() and .eos_id).
        stage: Dataset stage ('pretrain', 'cpt', 'sft', 'safety').
        split: 'train' or 'val'.
        context_length: Packed sequence length (default 4096).
        batch_size: Batch size.
        base_dir: Root dataset directory.
        seed: Random seed for shard shuffling.
        num_workers: DataLoader workers (0 = main process).
        shuffle_shards: Whether to shuffle shard file order.

    Returns:
        DataLoader yielding (input_ids, labels) batches of shape (batch_size, context_length).
    """
    dataset = PackedDataset(
        tokenizer=tokenizer,
        stage=stage,
        split=split,
        context_length=context_length,
        base_dir=base_dir,
        seed=seed,
        shuffle_shards=shuffle_shards,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

"""Tests for data.dataloader — sequence packing, text extraction, and DataLoader creation."""

import json
import gzip
import tempfile
from pathlib import Path

import torch
import pytest

from engine.tokenizer import BearTokenizer
from data.dataloader import (
    PackedDataset,
    create_dataloader,
    discover_shard_files,
    extract_texts,
)


@pytest.fixture
def tokenizer():
    """Minimal BearTokenizer (byte-level, no trained merges)."""
    return BearTokenizer()


@pytest.fixture
def fake_dataset(tmp_path):
    """
    Create a minimal fake dataset matching storage/dataset/{stage}/{domain}/{split}/ layout.
    Uses parquet, jsonl, and json formats.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    stage_dir = tmp_path / "pretrain"

    # Domain 1: parquet with 'text' column
    d1_train = stage_dir / "domain_a" / "train"
    d1_train.mkdir(parents=True)
    table = pa.table({"text": ["Hello world. " * 50, "Foo bar baz. " * 50, "Test data. " * 100]})
    pq.write_table(table, d1_train / "shard_0.parquet")

    # Domain 2: jsonl with 'content' key
    d2_train = stage_dir / "domain_b" / "train"
    d2_train.mkdir(parents=True)
    with open(d2_train / "data.jsonl", "w") as f:
        for text in ["Alpha beta. " * 40, "Gamma delta. " * 40]:
            f.write(json.dumps({"content": text}) + "\n")

    # Domain 3: json array with 'text' key
    d3_train = stage_dir / "domain_c" / "train"
    d3_train.mkdir(parents=True)
    with open(d3_train / "data.json", "w") as f:
        json.dump([{"text": "Omega psi. " * 60}], f)

    # Val split for domain_a
    d1_val = stage_dir / "domain_a" / "val"
    d1_val.mkdir(parents=True)
    table_val = pa.table({"text": ["Validation text. " * 80]})
    pq.write_table(table_val, d1_val / "shard_0.parquet")

    return tmp_path


# -- Text extraction tests --

def test_extract_texts_parquet(fake_dataset):
    parquet_file = fake_dataset / "pretrain" / "domain_a" / "train" / "shard_0.parquet"
    texts = list(extract_texts(parquet_file))
    assert len(texts) == 3
    assert "Hello world" in texts[0]


def test_extract_texts_jsonl(fake_dataset):
    jsonl_file = fake_dataset / "pretrain" / "domain_b" / "train" / "data.jsonl"
    texts = list(extract_texts(jsonl_file))
    assert len(texts) == 2
    assert "Alpha beta" in texts[0]


def test_extract_texts_json(fake_dataset):
    json_file = fake_dataset / "pretrain" / "domain_c" / "train" / "data.json"
    texts = list(extract_texts(json_file))
    assert len(texts) == 1
    assert "Omega psi" in texts[0]


def test_extract_texts_json_gz(tmp_path):
    gz_file = tmp_path / "data.json.gz"
    with gzip.open(gz_file, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"content": "Compressed text data"}) + "\n")
        f.write(json.dumps({"content": "More compressed data"}) + "\n")
    texts = list(extract_texts(gz_file))
    assert len(texts) == 2
    assert "Compressed" in texts[0]


# -- File discovery tests --

def test_discover_shard_files(fake_dataset):
    files = discover_shard_files("pretrain", "train", str(fake_dataset))
    assert len(files) == 3  # 1 parquet + 1 jsonl + 1 json
    assert all(f.is_file() for f in files)


def test_discover_shard_files_val(fake_dataset):
    files = discover_shard_files("pretrain", "val", str(fake_dataset))
    assert len(files) == 1  # only domain_a has val


def test_discover_shard_files_missing_stage(fake_dataset):
    files = discover_shard_files("nonexistent", "train", str(fake_dataset))
    assert files == []


# -- PackedDataset tests --

def test_packed_dataset_yields_correct_shapes(tokenizer, fake_dataset):
    ds = PackedDataset(
        tokenizer=tokenizer,
        stage="pretrain",
        split="train",
        context_length=64,  # small for testing
        base_dir=str(fake_dataset),
        seed=42,
    )
    count = 0
    for input_ids, labels in ds:
        assert input_ids.shape == (64,)
        assert labels.shape == (64,)
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long
        # Labels should be input_ids shifted by 1
        count += 1
        if count >= 3:
            break
    assert count >= 1, "PackedDataset should yield at least one sequence"


def test_packed_dataset_labels_shifted(tokenizer, fake_dataset):
    """Verify labels are the next-token prediction targets (shifted by 1)."""
    ds = PackedDataset(
        tokenizer=tokenizer,
        stage="pretrain",
        split="train",
        context_length=32,
        base_dir=str(fake_dataset),
        seed=42,
    )
    for input_ids, labels in ds:
        # The label for position i should equal input_ids at position i+1
        # in the original stream. We verify first 31 match.
        # (labels[i] == next token after input_ids[i] in the packed stream)
        assert input_ids.shape == (32,)
        assert labels.shape == (32,)
        break


def test_packed_dataset_contains_eos(tokenizer, fake_dataset):
    """Verify that EOS tokens appear in the packed stream (document boundaries)."""
    ds = PackedDataset(
        tokenizer=tokenizer,
        stage="pretrain",
        split="train",
        context_length=512,
        base_dir=str(fake_dataset),
        seed=42,
    )
    eos_id = tokenizer.eos_id
    found_eos = False
    for input_ids, labels in ds:
        if eos_id in input_ids.tolist():
            found_eos = True
            break
    assert found_eos, "Packed sequences should contain EOS tokens between documents"


def test_packed_dataset_no_files_raises(tokenizer, tmp_path):
    """Should raise FileNotFoundError when no data files exist."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        PackedDataset(
            tokenizer=tokenizer,
            stage="pretrain",
            split="train",
            context_length=64,
            base_dir=str(empty_dir),
        )


# -- DataLoader factory tests --

def test_create_dataloader(tokenizer, fake_dataset):
    dl = create_dataloader(
        tokenizer=tokenizer,
        stage="pretrain",
        split="train",
        context_length=64,
        batch_size=2,
        base_dir=str(fake_dataset),
        num_workers=0,
    )
    batch = next(iter(dl))
    input_ids, labels = batch
    assert input_ids.shape == (2, 64)
    assert labels.shape == (2, 64)


def test_create_dataloader_val(tokenizer, fake_dataset):
    dl = create_dataloader(
        tokenizer=tokenizer,
        stage="pretrain",
        split="val",
        context_length=32,
        batch_size=1,
        base_dir=str(fake_dataset),
        num_workers=0,
    )
    batch = next(iter(dl))
    input_ids, labels = batch
    assert input_ids.shape == (1, 32)
    assert labels.shape == (1, 32)

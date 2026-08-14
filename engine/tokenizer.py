import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

try:
    import regex
    HAS_REGEX = True
except ImportError:
    import re as regex
    HAS_REGEX = False

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    tiktoken = None
    HAS_TIKTOKEN = False

# Bear AI Special Tokens (Compatible with Kimi K3 XTML Chat Standard)
DEFAULT_SPECIAL_TOKENS_LIST = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|end_of_msg|>",
    "<|open|>",
    "<|close|>",
    "<|sep|>",
    "[start_header_id]",
    "[end_header_id]",
    "[EOT]",
    "<|media_begin|>",
    "<|media_content|>",
    "<|media_end|>",
    "<|pad|>",
    "<|unk|>",
]


def create_special_tokens(offset: int = 59986) -> Dict[str, int]:
    """Create special token mapping positioned immediately after base vocabulary merges."""
    return {tok: offset + i for i, tok in enumerate(DEFAULT_SPECIAL_TOKENS_LIST)}


BEAR_SPECIAL_TOKENS = create_special_tokens(59986)

# Bear Kimi K3 Regex Pattern for multi-language & multi-domain tokenization (No backtracking)
BEAR_PAT_STR = "|".join([
    r"[\p{Han}]+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
]) if HAS_REGEX else r"\w+|\s+|[^\w\s]+"


def _get_pairs(word: List[bytes]) -> set:
    """Get all adjacent byte pairs in a sequence of byte tokens."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BearTokenizer:
    """
    Mesosfer Bear AI Custom Tokenizer.
    
    Engineered based on Kimi K3 BPE architecture with custom optimizations for:
    - Multi-domain corpus (General, Code, Terminal/PowerShell/Bash, Science, Math, CoT).
    - High-performance Batch Encoding & Decoding.
    - Native BPE Training pipeline from local datasets.
    - XTML Chat Template Markup rendering.
    """

    def __init__(
        self,
        ranks: Optional[Dict[bytes, int]] = None,
        special_tokens: Optional[Dict[str, int]] = None,
        pat_str: str = BEAR_PAT_STR,
        use_tiktoken: bool = True,
    ):
        self.pat_str = pat_str
        self.compiled_pat = regex.compile(pat_str)

        # Base byte-to-token ranks mapping (BPE merge ranks)
        self.ranks: Dict[bytes, int] = ranks or {bytes([b]): b for b in range(256)}
        self.decoder: Dict[int, bytes] = {v: k for k, v in self.ranks.items()}

        # Special tokens
        self.special_tokens = special_tokens or create_special_tokens(len(self.ranks))
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        # Core Special Token IDs
        self.bos_token = "<|begin_of_text|>"
        self.eos_token = "<|end_of_text|>"
        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"

        self.bos_id = self.special_tokens.get(self.bos_token)
        self.eos_id = self.special_tokens.get(self.eos_token)
        self.pad_id = self.special_tokens.get(self.pad_token)
        self.unk_id = self.special_tokens.get(self.unk_token)

        # ponytail: tiktoken Rust backend — zero-conversion bridge from self.ranks
        # Falls back to pure-Python _bpe_encode_piece if tiktoken not installed
        self._tiktoken_enc = None
        if use_tiktoken and HAS_TIKTOKEN:
            self._tiktoken_enc = self._build_tiktoken_encoding()

    def _build_tiktoken_encoding(self):
        """Build a tiktoken.Encoding from our existing ranks — exact same data, Rust speed."""
        return tiktoken.Encoding(
            name="bear",
            pat_str=self.pat_str,
            mergeable_ranks=self.ranks,
            special_tokens=self.special_tokens,
        )

    @property
    def vocab_size(self) -> int:
        base_size = len(self.ranks)
        max_special = max(self.special_tokens.values(), default=-1)
        return max(base_size, max_special + 1)

    def _bpe_encode_piece(self, piece_bytes: bytes) -> List[int]:
        """Encode a single regex chunk of bytes using BPE merge ranks."""
        if piece_bytes in self.ranks:
            return [self.ranks[piece_bytes]]

        word: List[bytes] = [bytes([b]) for b in piece_bytes]
        pairs = _get_pairs(word)

        if not pairs:
            return [self.ranks.get(b, self.unk_id) for b in word]

        while True:
            # Find the pair with the lowest rank index
            min_pair = min(pairs, key=lambda pair: self.ranks.get(pair[0] + pair[1], float("inf")))
            merged_bytes = min_pair[0] + min_pair[1]
            if merged_bytes not in self.ranks:
                break

            new_word: List[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == min_pair[0] and word[i + 1] == min_pair[1]:
                    new_word.append(merged_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        return [self.ranks.get(b, self.unk_id) for b in word]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        allow_special: bool = False
    ) -> List[int]:
        """
        Encode text into a list of token IDs.
        Uses tiktoken Rust backend when available, falls back to pure-Python BPE.
        """
        tokens: List[int] = []

        if add_special_tokens and self.bos_id is not None:
            tokens.append(self.bos_id)

        # Fast path: tiktoken Rust backend
        if self._tiktoken_enc is not None:
            allowed = set(self.special_tokens.keys()) if allow_special else set()
            tokens.extend(self._tiktoken_enc.encode(text, allowed_special=allowed, disallowed_special=()))
        else:
            # Slow path: pure-Python BPE
            for match in self.compiled_pat.finditer(text):
                piece = match.group(0)
                if allow_special and piece in self.special_tokens:
                    tokens.append(self.special_tokens[piece])
                else:
                    piece_bytes = piece.encode("utf-8")
                    tokens.extend(self._bpe_encode_piece(piece_bytes))

        if add_special_tokens and self.eos_id is not None:
            tokens.append(self.eos_id)

        return tokens

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
        allow_special: bool = False
    ) -> List[List[int]]:
        """Encode a batch of text strings into token ID lists."""
        return [self.encode(t, add_special_tokens=add_special_tokens, allow_special=allow_special) for t in texts]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode a list of token IDs back into text.
        Uses tiktoken Rust backend when available, falls back to pure-Python.
        """
        # Fast path: tiktoken handles non-special decode natively
        if self._tiktoken_enc is not None and not skip_special_tokens:
            return self._tiktoken_enc.decode(token_ids)

        # Slow path / skip_special_tokens: pure-Python
        byte_chunks: List[bytes] = []
        for tid in token_ids:
            if tid in self.inverse_special_tokens:
                if not skip_special_tokens:
                    byte_chunks.append(self.inverse_special_tokens[tid].encode("utf-8"))
            elif tid in self.decoder:
                byte_chunks.append(self.decoder[tid])
            else:
                if 0 <= tid <= 255:
                    byte_chunks.append(bytes([tid]))
                else:
                    byte_chunks.append(b"")

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def decode_batch(self, batch_ids: List[List[int]], skip_special_tokens: bool = False) -> List[str]:
        """Decode a batch of token ID lists back into text strings."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        thinking: bool = True,
        tokenize: bool = True
    ) -> Union[str, List[int]]:
        """
        Render conversation messages into Bear XTML Chat Format:
        <|open|>message role="system"<|close|>System prompt...<|end_of_msg|>
        <|open|>message role="user"<|close|>User prompt...<|end_of_msg|>
        <|open|>message role="assistant" thinking="max"<|close|>
        """
        formatted_text = ""
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_text += f"<|open|>message role=\"{role}\"<|close|>{content}<|end_of_msg|>"

        if add_generation_prompt:
            formatted_text += "<|open|>message role=\"assistant\""
            if thinking:
                formatted_text += " thinking=\"max\""
            formatted_text += "<|close|>"

        if not tokenize:
            return formatted_text

        return self.encode(formatted_text, add_special_tokens=True, allow_special=True)

    def save(self, filepath: str):
        """Save vocabulary and config to JSON file."""
        data = {
            "name": "BearTokenizer",
            "special_tokens": self.special_tokens,
            "pat_str": self.pat_str,
            "ranks": {base64.b64encode(k).decode("ascii"): v for k, v in self.ranks.items()}
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BearTokenizer":
        """Load BearTokenizer from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        ranks = {base64.b64decode(k.encode("ascii")): v for k, v in data["ranks"].items()}
        special_tokens = data.get("special_tokens", BEAR_SPECIAL_TOKENS)
        pat_str = data.get("pat_str", BEAR_PAT_STR)
        return cls(ranks=ranks, special_tokens=special_tokens, pat_str=pat_str)

    from_file = load

    @classmethod
    def train_from_iterator(
        cls,
        iterator,
        vocab_size: int = 16384,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, int]] = None,
        pat_str: str = BEAR_PAT_STR,
        verbose: bool = True,
    ) -> "BearTokenizer":
        """
        Train a BPE vocabulary from an iterator/generator of text strings.
        Uses optimized inverted-index merge updates for high-performance training.
        """
        import time
        import heapq
        from collections import defaultdict, Counter

        t0 = time.time()
        if verbose:
            print(f"=== [BearTokenizer] Starting BPE Training (Target Vocab: {vocab_size}) ===", flush=True)

        compiled_pat = regex.compile(pat_str)
        ranks: Dict[bytes, int] = {bytes([b]): b for b in range(256)}
        next_rank = 256

        # Step 1: Pre-tokenize and count word byte frequencies
        if verbose:
            print("Step 1/3: Extracting regex word tokens from corpus...", flush=True)
        word_counts: Dict[Tuple[bytes, ...], int] = Counter()
        total_chars = 0
        total_chunks = 0

        for text in iterator:
            if not text:
                continue
            total_chars += len(text)
            for match in compiled_pat.finditer(text):
                piece = match.group(0)
                piece_bytes = piece.encode("utf-8")
                if len(piece_bytes) > 0:
                    word_tuple = tuple(bytes([b]) for b in piece_bytes)
                    word_counts[word_tuple] += 1
                    total_chunks += 1

        if verbose:
            print(f"  Processed {total_chars:,} chars ({total_chunks:,} token chunks, {len(word_counts):,} unique words)", flush=True)

        # Step 2: Build pair frequency table and inverted index
        if verbose:
            print("Step 2/3: Building pair frequency table and inverted index...", flush=True)
        pair_counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
        pair_to_words: Dict[Tuple[bytes, bytes], set] = defaultdict(set)

        for word_tuple, count in word_counts.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pair_counts[pair] += count
                pair_to_words[pair].add(word_tuple)

        # Initialize max-heap with (-count, pair)
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        # Step 3: Iterative BPE Merge with fast inverted index updates
        if verbose:
            print("Step 3/3: Running iterative BPE merge loop...", flush=True)

        num_special = len(special_tokens) if special_tokens is not None else len(DEFAULT_SPECIAL_TOKENS_LIST)
        target_base_size = vocab_size - num_special if vocab_size > num_special else vocab_size
        target_merges = target_base_size - 256
        merges_done = 0
        log_interval = max(500, target_merges // 10) if target_merges > 0 else 500

        while len(ranks) < target_base_size and heap:
            neg_count, best_pair = heapq.heappop(heap)
            current_count = pair_counts.get(best_pair, 0)
            if -neg_count != current_count or current_count == 0:
                continue  # Stale entry from heap

            if current_count < min_frequency:
                if verbose:
                    print(f"  Reached min frequency threshold ({current_count} < {min_frequency}). Stopping.", flush=True)
                break

            merged_bytes = best_pair[0] + best_pair[1]
            ranks[merged_bytes] = next_rank
            next_rank += 1
            merges_done += 1

            if verbose and merges_done % log_interval == 0:
                print(f"  [Merge {merges_done}/{target_merges}] Base Vocab: {len(ranks):,} | Best Pair: {best_pair!r} ({current_count:,} occurrences)", flush=True)

            # Update only the words that contain best_pair
            affected_words = list(pair_to_words.get(best_pair, set()))
            modified_pairs = set()

            for word in affected_words:
                if word not in word_counts:
                    continue
                count = word_counts.pop(word)

                # Remove old pairs of this word
                for i in range(len(word) - 1):
                    p = (word[i], word[i + 1])
                    pair_counts[p] -= count
                    modified_pairs.add(p)
                    if p in pair_to_words:
                        pair_to_words[p].discard(word)
                        if not pair_to_words[p]:
                            pair_to_words.pop(p, None)

                # Construct new word with merged bytes
                new_word: List[bytes] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        new_word.append(merged_bytes)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                new_word_tuple = tuple(new_word)
                word_counts[new_word_tuple] = word_counts.get(new_word_tuple, 0) + count

                # Add new pairs of new_word
                for i in range(len(new_word_tuple) - 1):
                    p = (new_word_tuple[i], new_word_tuple[i + 1])
                    pair_counts[p] += count
                    modified_pairs.add(p)
                    pair_to_words[p].add(new_word_tuple)

            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

            # Update heap once per distinct modified pair
            for p in modified_pairs:
                c = pair_counts.get(p, 0)
                if c <= 0:
                    pair_counts.pop(p, None)
                else:
                    heapq.heappush(heap, (-c, p))

        elapsed = time.time() - t0
        spec_tokens = special_tokens or create_special_tokens(len(ranks))
        if verbose:
            print(f"=== [BearTokenizer] Training Complete in {elapsed:.2f}s! ===", flush=True)
            print(f"  Base Vocab Size: {len(ranks):,} | Special Tokens: {len(spec_tokens)} | Total Vocab Size: {max(len(ranks), max(spec_tokens.values(), default=0) + 1):,}", flush=True)

        return cls(ranks=ranks, special_tokens=spec_tokens, pat_str=pat_str)

    @classmethod
    def train_from_files(
        cls,
        files: List[str],
        vocab_size: int = 16384,
        min_frequency: int = 2,
        max_bytes_per_file: int = 10 * 1024 * 1024,
    ) -> "BearTokenizer":
        """
        Train a BPE vocabulary from a list of raw text files.
        """
        def file_text_generator():
            for file_path in files:
                if not os.path.exists(file_path):
                    continue
                print(f"Reading corpus file: {file_path}")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    while True:
                        chunk = f.read(max_bytes_per_file)
                        if not chunk:
                            break
                        yield chunk

        return cls.train_from_iterator(
            file_text_generator(),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )


# Alias for backward compatibility
KimiK3Tokenizer = BearTokenizer


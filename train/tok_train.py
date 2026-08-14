import os
import sys
import glob
import argparse
from pathlib import Path
from typing import Generator, List, Dict, Optional

import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.tokenizer import BearTokenizer, BEAR_SPECIAL_TOKENS, BEAR_PAT_STR


from data.dataloader import extract_texts


def corpus_streamer(dataset_dir: str = "storage/dataset", sample_limit_per_domain: int = 5000) -> Generator[str, None, None]:
    """
    Stream combined multi-domain text samples from dataset storage.
    Traverses all stages (pretrain, cpt, sft, safety) and domain folders in storage/dataset.
    """
    base = Path(dataset_dir)
    if not base.exists():
        print(f"[Warning] Dataset directory not found: {dataset_dir}", flush=True)
        return

    # Find all domain train directories
    domain_dirs: List[Tuple[str, Path]] = []
    for stage_dir in sorted(base.iterdir()):
        if not stage_dir.is_dir():
            continue
        for dom_dir in sorted(stage_dir.iterdir()):
            if not dom_dir.is_dir():
                continue
            train_dir = dom_dir / "train"
            target_dir = train_dir if train_dir.exists() else dom_dir
            domain_label = f"{stage_dir.name}/{dom_dir.name}"
            domain_dirs.append((domain_label, target_dir))

    if not domain_dirs:
        print(f"[Warning] No domain directories found in {dataset_dir}", flush=True)
        return

    print(f"Combining {len(domain_dirs)} domain sources into unified corpus:", flush=True)
    for label, _ in domain_dirs:
        print(f"  - {label}", flush=True)

    for label, domain_path in domain_dirs:
        # Find all data files in domain directory
        data_files = [
            f for f in sorted(domain_path.rglob("*"))
            if f.is_file() and not f.name.endswith(".tmp") and f.suffix in (".parquet", ".json", ".jsonl", ".gz")
        ]

        if not data_files:
            continue

        print(f"  -> Streaming [{label}]: Up to {sample_limit_per_domain:,} samples across {len(data_files)} file(s)...", flush=True)
        count = 0

        for file_path in data_files:
            try:
                for text in extract_texts(file_path):
                    if text and text.strip():
                        yield text.strip()
                        count += 1
                        if count >= sample_limit_per_domain:
                            break
            except Exception as e:
                print(f"    [Warning] Failed reading {file_path.name}: {e}", flush=True)

            if count >= sample_limit_per_domain:
                break
        print(f"     Collected {count:,} samples from [{label}]", flush=True)


def run_benchmark(tokenizer: BearTokenizer):
    """Benchmark compression and roundtrip lossless verification across all combined domains."""
    print("\n" + "=" * 65, flush=True)
    print("=== [Verification Benchmark & Lossless Reconstruction] ===", flush=True)
    print("=" * 65, flush=True)

    test_samples = {
        "Indonesian (Umum)": "Mesosfer Bear AI adalah arsitektur model bahasa mutakhir yang dilatih khusus untuk efisiensi tinggi.",
        "English (General)": "Byte Pair Encoding constructs subword vocabularies based on frequency statistics of byte merges.",
        "Python Code": "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        "PowerShell & Terminal": "Get-ChildItem -Path ./storage/dataset -Recurse | Where-Object { $_.Length -gt 1024 } | Select-Object Name, Length",
        "Bash & Linux Admin": "find /var/log -type f -name '*.log' -mtime +30 -exec gzip {} \\; && systemctl restart nginx",
        "Git CLI & Workflow": "git checkout -b feat/unified-tokenizer && git commit -am 'feat: unified domain tokens' && git push -u origin main",
        "Docker & DevOps": "docker run -d --name bear-srv -p 8080:80 -v /app/data:/data --restart always nginx:alpine",
        "LaTeX / Math": r"\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}, \quad \sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
        "Chain of Thought": "Step 1: Parse the user query. Step 2: Formulate hypothesis. Step 3: Verify edge cases before computing result.",
    }

    all_passed = True
    for label, text in test_samples.items():
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        bytes_len = len(text.encode("utf-8"))
        tokens_len = len(token_ids)
        compression = bytes_len / max(tokens_len, 1)
        passed = (decoded == text)
        if not passed:
            all_passed = False

        status = "PASSED [Lossless]" if passed else "FAILED [Mismatch]"
        print(f"[{label}] -> {status}", flush=True)
        print(f"  Raw Bytes: {bytes_len} B | Tokens: {tokens_len} | Ratio: {compression:.2f} bytes/token", flush=True)
        print(f"  First 8 Token IDs: {token_ids[:8]}", flush=True)

    print("-" * 65, flush=True)
    # Test XTML Chat Template
    chat_messages = [
        {"role": "system", "content": "You are Bear AI assistant trained on unified multi-domain knowledge."},
        {"role": "user", "content": "Halo! Tolong jelaskan cara kerja tokenizer BPE dan berikan contoh script PowerShell."}
    ]
    chat_markup = tokenizer.apply_chat_template(chat_messages, thinking=True, tokenize=False)
    chat_token_ids = tokenizer.apply_chat_template(chat_messages, thinking=True, tokenize=True)
    chat_decoded = tokenizer.decode(chat_token_ids, skip_special_tokens=False)

    print("[XTML Chat Template Test]", flush=True)
    print(f"  Formatted Prompt:\n  {chat_markup}", flush=True)
    print(f"  Encoded Token Count: {len(chat_token_ids)}")
    print(f"  Decoded with Special Tokens: {'PASSED' if '<|open|>' in chat_decoded else 'FAILED'}", flush=True)

    print("=" * 65, flush=True)
    if all_passed:
        print("[SUCCESS] All Tokenizer Verification & Compression Benchmarks PASSED!", flush=True)
    else:
        print("[WARNING] Some reconstruction checks did not match perfectly.", flush=True)
    print("=" * 65, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train Bear AI Tokenizer on Unified Multi-Domain Corpus")
    parser.add_argument("--dataset-dir", type=str, default="storage/dataset", help="Path to multi-domain dataset directory")
    parser.add_argument("--vocab-size", type=int, default=60000, help="Target vocabulary size (default: 60000)")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum merge frequency threshold")
    parser.add_argument("--sample-limit", type=int, default=2500, help="Max samples per domain (default: 2500)")
    parser.add_argument("--save-path", type=str, default="storage/tokenizer/bear_tokenizer.json", help="Path to save trained JSON artifact")
    args = parser.parse_args()

    print(f"=== Mesosfer Bear AI: Unified Multi-Domain Tokenizer Training ===", flush=True)
    print(f"Dataset Dir     : {args.dataset_dir}", flush=True)
    print(f"Target Vocab    : {args.vocab_size:,}", flush=True)
    print(f"Min Frequency   : {args.min_frequency}", flush=True)
    print(f"Sample Limit/Dom: {args.sample_limit:,} samples/domain", flush=True)
    print(f"Save Path       : {args.save_path}\n", flush=True)

    # 1. Stream unified training texts from all combined domains
    stream = corpus_streamer(args.dataset_dir, sample_limit_per_domain=args.sample_limit)

    # 2. Train BearTokenizer with Kimi K3 architecture (Target: 60,000 vocab)
    tokenizer = BearTokenizer.train_from_iterator(
        stream,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=None,
        pat_str=BEAR_PAT_STR,
        verbose=True,
    )

    # 3. Save trained tokenizer
    print(f"\nSaving trained tokenizer to: {args.save_path} ...", flush=True)
    tokenizer.save(args.save_path)
    print(f"Artifact successfully saved! File size: {os.path.getsize(args.save_path):,} bytes", flush=True)

    # 4. Run Benchmark & Lossless Reconstruction Verification
    run_benchmark(tokenizer)


if __name__ == "__main__":
    main()

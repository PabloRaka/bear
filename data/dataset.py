"""
Mesosfer Bear AI - Real Multi-Domain Dataset Downloader
Structured into 4 Training Stages:
  1. Pretrain     (Bahasa 50%, Code 20%, Math 20%, Terminal 10%)
  2. CPT          (Bahasa 10%, Code 40%, Math 30%, Terminal 20%)
  3. SFT          (Indonesian & English Chat 80%, Instruction 10%, Tool Calls 10%)
  4. Safety Train (Crime / Exploit / Harm Refusal & Guardrails 100%)
"""

import os
import sys
import json
import argparse
import urllib.request
import fnmatch
from pathlib import Path
from typing import Dict, List, Any, Optional

# Verified Open-Access Hugging Face Repositories per Stage (No auth token required)
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pretrain": {
        "title": "Stage 1: Pre-training (Foundational Language & General Knowledge)",
        "domains": {
            "bahasa_umum_id": {
                "name": "Bahasa Indonesia (Wikipedia ID)",
                "weight": 0.25,
                "repo": "wikimedia/wikipedia",
                "pattern": "20231101.id/train-*.parquet",
                "max_production_shards": 3,
                "description": "Indonesian Wikipedia full encyclopedic raw articles"
            },
            "bahasa_umum_en": {
                "name": "English General (ClimbMix-400B NVIDIA/Karpathy)",
                "weight": 0.25,
                "repo": "karpathy/climbmix-400b-shuffle",
                "pattern": "shard_*.parquet",
                "max_production_shards": 5,
                "description": "High-quality curated pre-training corpus by NVIDIA & Andrej Karpathy"
            },
            "code_python": {
                "name": "Code: Python (CodeParrot Clean)",
                "weight": 0.03,
                "repo": "codeparrot/codeparrot-clean-valid",
                "pattern": "*.json.gz",
                "max_production_shards": 1,
                "description": "Raw Python repository source files from GitHub"
            },
            "code_typescript": {
                "name": "Code: TypeScript (TS GitHub)",
                "weight": 0.03,
                "repo": "petrpan26/typescript-code",
                "pattern": "data/train-*.parquet",
                "max_production_shards": 6,
                "description": "Raw TypeScript (.ts, .tsx) codebase source files"
            },
            "code_javascript": {
                "name": "Code: JavaScript (CodeSearchNet JS)",
                "weight": 0.03,
                "repo": "code_search_net",
                "pattern": "javascript/train-*.parquet",
                "max_production_shards": 1,
                "description": "JavaScript (.js) source files and modules"
            },
            "code_php": {
                "name": "Code: PHP (CodeSearchNet PHP)",
                "weight": 0.03,
                "repo": "code_search_net",
                "pattern": "php/train-*.parquet",
                "max_production_shards": 1,
                "description": "PHP (.php) backend and web application code"
            },
            "code_cpp": {
                "name": "Code: C++ (ArXiv C++ Research)",
                "weight": 0.03,
                "repo": "AlgorithmicResearchGroup/arxiv_cplusplus_research_code",
                "pattern": "data/train-*.parquet",
                "max_production_shards": 5,
                "description": "C++ (.cpp, .hpp) algorithms, scientific libraries, and system code"
            },
            "code_c": {
                "name": "Code: C (Torvalds Linux Kernel C)",
                "weight": 0.025,
                "repo": "kye/all-torvalds-c-code-1",
                "pattern": "data/train-*.parquet",
                "max_production_shards": 2,
                "description": "C (.c, .h) systems programming and kernel algorithms"
            },
            "code_csharp": {
                "name": "Code: C# (Microsoft LCC CSharp)",
                "weight": 0.025,
                "repo": "microsoft/LCC_csharp",
                "pattern": "data/train-*.parquet",
                "max_production_shards": 4,
                "description": "C# (.cs) enterprise and application codebases"
            },
            "matematika": {
                "name": "Matematika (OpenWebMath)",
                "weight": 0.20,
                "repo": "open-web-math/open-web-math",
                "pattern": "data/train-*.parquet",
                "max_production_shards": 5,
                "description": "Mathematical documents, LaTeX proofs, equations, and reasoning"
            },
            "terminal": {
                "name": "Terminal (PowerShell & Shell Scripts)",
                "weight": 0.10,
                "repo": "SaeedRahmani/codeparrot_github_code_powershell",
                "pattern": "data/train-*.parquet",
                "max_production_shards": 3,
                "description": "PowerShell scripts, system administration commands, and CLI automation"
            },
        }
    },
    "cpt": {
        "title": "Stage 2: Continued Pre-Training (Domain Specialization)",
        "domains": {
            "bahasa_umum": {
                "name": "Bahasa Umum (ArXiv Scientific)",
                "weight": 0.10,
                "repo": "ccdv/arxiv-classification",
                "pattern": "data/train-*.parquet",
                "max_production_shards": 4,
                "description": "High-density scientific and technical literature"
            },
            "code_multilang": {
                "name": "Code: Multi-Language Deep Engineering",
                "weight": 0.40,
                "repo": "code_search_net",
                "pattern": "all/train-*.parquet",
                "max_production_shards": 2,
                "description": "Multi-language deep source code (Python, JS, PHP, Go, Java, Ruby)"
            },
            "matematika": {
                "name": "Matematika: OpenWebMath Reasoning",
                "weight": 0.30,
                "repo": "open-web-math/open-web-math",
                "pattern": "data/train-*.parquet",
                "max_production_shards": 5,
                "description": "Mathematical proofs, formal expressions, and symbolic logic"
            },
            "terminal": {
                "name": "Terminal: PowerShell & CLI Admin",
                "weight": 0.20,
                "repo": "SaeedRahmani/codeparrot_github_code_powershell",
                "pattern": "data/train-*.parquet",
                "max_production_shards": 3,
                "description": "Advanced systems automation, scripting, and shell operations"
            },
        }
    },
    "sft": {
        "title": "Stage 3: Supervised Fine-Tuning (Instruction & Dialogue)",
        "domains": {
            "percakapan_id": {
                "name": "Percakapan Bahasa Indonesia (Alpaca-GPT4-ID)",
                "weight": 0.40,
                "repo": "FreedomIntelligence/alpaca-gpt4-indonesian",
                "pattern": "*.json",
                "max_production_shards": 1,
                "description": "Indonesian natural conversations and instruction responses"
            },
            "percakapan_en": {
                "name": "Percakapan English (UltraChat 200k)",
                "weight": 0.40,
                "repo": "HuggingFaceH4/ultrachat_200k",
                "pattern": "data/train_sft-*.parquet",
                "max_production_shards": 3,
                "description": "Multi-turn conversational dialogue and informative interactions"
            },
            "instruksi": {
                "name": "Instruksi & Penalaran (Open-Platypus)",
                "weight": 0.10,
                "repo": "garage-bAInd/Open-Platypus",
                "pattern": "data/train-*.parquet",
                "max_production_shards": 1,
                "description": "STEM, logic, and chain-of-thought instruction-following tasks"
            },
            "tooling_calls": {
                "name": "Tooling & Function Calling (Glaive v2)",
                "weight": 0.10,
                "repo": "glaiveai/glaive-function-calling-v2",
                "pattern": "*.json",
                "max_production_shards": 1,
                "description": "Agent tool invocation, JSON arguments, and execution schemas"
            },
        }
    },
    "safety": {
        "title": "Stage 4: Safety & Guardrails (Harm & Crime Refusal)",
        "domains": {
            "safety_pku": {
                "name": "Penolakan Kejahatan & Harm Refusal (PKU-SafeRLHF)",
                "weight": 0.50,
                "repo": "PKU-Alignment/PKU-SafeRLHF",
                "pattern": "data/Alpaca-7B/train.jsonl",
                "max_production_shards": 1,
                "description": "Refusal of illegal acts, cyberattacks, weapons, privacy violations, and harm"
            },
            "safety_anthropic": {
                "name": "Red-Teaming Harmlessness (Anthropic HH-RLHF)",
                "weight": 0.30,
                "repo": "Anthropic/hh-rlhf",
                "pattern": "harmless-base/train.jsonl.gz",
                "max_production_shards": 1,
                "description": "Red-teaming dialogues and harmless response alignment"
            },
            "safety_jailbreak": {
                "name": "Adversarial Jailbreak Defense (JailbreakHub)",
                "weight": 0.20,
                "repo": "walledai/JailbreakHub",
                "pattern": "data/train-*.parquet",
                "max_production_shards": 1,
                "description": "Adversarial prompts, DAN jailbreak defenses, and robust refusals"
            },
        }
    }
}


def get_verified_shards(repo_id: str, pattern: str = "*", limit: Optional[int] = 50, max_retries: int = 3) -> List[str]:
    """
    Fetch and filter available shard files from Hugging Face Hub API matching pattern with retry.
    If limit is None or <= 0, returns all matching shards.
    """
    import time
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    siblings = []
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                siblings = data.get("siblings", [])
                break
        except Exception as e:
            if attempt == max_retries:
                print(f"    [WARN] Failed to query HF API for {repo_id} (Attempt {attempt}/{max_retries}): {e}")
                return []
            time.sleep(2 * attempt)

    valid_extensions = (".parquet", ".json", ".jsonl", ".csv", ".json.gz", ".jsonl.gz", ".gz")
    matching_files = []
    
    for s in siblings:
        rfilename = s.get("rfilename", "")
        if not rfilename.endswith(valid_extensions):
            continue
        if fnmatch.fnmatch(rfilename, pattern) or fnmatch.fnmatch(Path(rfilename).name, pattern):
            matching_files.append(rfilename)

    # If pattern was overly specific and found nothing, fallback to all valid data files in repo
    if not matching_files:
        matching_files = [
            s["rfilename"] for s in siblings
            if s.get("rfilename", "").endswith(valid_extensions)
        ]

    if limit is not None and limit > 0:
        return matching_files[:limit]
    return matching_files


def download_shard(repo_id: str, rfilename: str, output_path: Path, max_retries: int = 3) -> bool:
    """
    Download a single file from Hugging Face with realtime download progress, atomic temp file, and auto-retry.
    """
    import time
    import shutil
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rfilename}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"    [EXISTS] {output_path.name} ({output_path.stat().st_size / (1024 * 1024):.2f} MB)")
        return True

    temp_output_path = output_path.with_name(f"{output_path.stem}_{os.getpid()}_{time.time_ns()}.tmp")
    print(f"    [DOWNLOADING] {rfilename} -> {output_path.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB buffer
                
                with open(temp_output_path, "wb") as out_file:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            percent = (downloaded / total) * 100
                            print(f"\r      [{percent:5.1f}%] {downloaded / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB", end="", flush=True)
                
                if total > 0 and downloaded < total:
                    raise IOError(f"Incomplete download stream: received {downloaded}/{total} bytes")

                print()
                time.sleep(0.05)
                if output_path.exists():
                    output_path.unlink()
                shutil.move(temp_output_path, output_path)
                return True
        except Exception as e:
            if temp_output_path.exists():
                try:
                    temp_output_path.unlink()
                except Exception:
                    pass
            print(f"\n    [WARN] Attempt {attempt}/{max_retries} failed for {rfilename}: {e}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                print(f"    [ERROR] Download failed permanently for {rfilename} after {max_retries} attempts.")
                return False
    return False


def print_stage_info():
    """Print the complete breakdown and weights of all dataset stages."""
    print("=" * 70)
    print(" MESOSFER BEAR AI - DATASET STAGES & DOMAIN DISTRIBUTION")
    print("=" * 70)
    for stage_key, stage_info in DATASET_REGISTRY.items():
        print(f"\n[{stage_key.upper()}] - {stage_info['title']}")
        for dom_key, dom_info in stage_info["domains"].items():
            pct = int(dom_info["weight"] * 100)
            print(f"  - {dom_info['name']} ({pct}%):")
            print(f"      Repo       : {dom_info['repo']}")
            print(f"      Pattern    : {dom_info['pattern']}")
            print(f"      Description: {dom_info['description']}")
    print("=" * 70)


def split_dataset_file(
    file_path: Path,
    train_dir: Path,
    val_dir: Path,
    val_ratio: float = 0.05,
    seed: int = 42,
    shuffle: bool = True,
):
    """
    Split a downloaded dataset shard into Train and Validation splits with structured deterministic shuffling.
    Supports Parquet, JSONL, JSON.GZ, and JSON files.
    Default: 95% Train, 5% Validation (val_ratio=0.05) with seed=42 to prevent gradient domain correlation and loss spikes.
    """
    import random
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = file_path.name
    train_file = train_dir / file_name
    val_file = val_dir / file_name
    
    if train_file.exists() and val_file.exists():
        print(f"    [SPLIT EXISTS] Skipping split for {file_name}")
        return train_file, val_file

    train_pct = int(round((1.0 - val_ratio) * 100))
    val_pct = int(round(val_ratio * 100))
    shuffle_info = f"with deterministic seed={seed}" if shuffle else "without shuffle"
    print(f"    [SPLITTING & SHUFFLING] {file_name} ({shuffle_info}) -> Train ({train_pct}%) / Val ({val_pct}%)...")

    try:
        # 1. Handle Parquet
        if file_path.name.endswith(".parquet"):
            import pyarrow.parquet as pq
            import pyarrow.compute as pc
            import pyarrow as pa

            table = pq.read_table(file_path)
            n_total = len(table)

            if shuffle and n_total > 1:
                rng = random.Random(seed)
                indices = list(range(n_total))
                rng.shuffle(indices)
                table = pc.take(table, pa.array(indices))

            n_val = max(1, int(n_total * val_ratio))
            n_train = n_total - n_val
            
            train_table = table.slice(0, n_train)
            val_table = table.slice(n_train, n_val)
            
            pq.write_table(train_table, train_file)
            pq.write_table(val_table, val_file)
            print(f"    [SHUFFLE & SPLIT COMPLETE] Train: {n_train:,} rows ({train_pct}%) | Val: {n_val:,} rows ({val_pct}%)")

        # 2. Handle JSON.GZ / JSONL.GZ
        elif file_path.name.endswith((".json.gz", ".jsonl.gz", ".gz")):
            import gzip
            with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            n_total = len(lines)

            if shuffle and n_total > 1:
                random.Random(seed).shuffle(lines)

            n_val = max(1, int(n_total * val_ratio))
            n_train = n_total - n_val
            
            with gzip.open(train_file, "wt", encoding="utf-8") as f_train:
                f_train.writelines(lines[:n_train])
            with gzip.open(val_file, "wt", encoding="utf-8") as f_val:
                f_val.writelines(lines[n_train:])
            print(f"    [SHUFFLE & SPLIT COMPLETE] Train: {n_train:,} lines ({train_pct}%) | Val: {n_val:,} lines ({val_pct}%)")

        # 3. Handle JSONL
        elif file_path.name.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            n_total = len(lines)

            if shuffle and n_total > 1:
                random.Random(seed).shuffle(lines)

            n_val = max(1, int(n_total * val_ratio))
            n_train = n_total - n_val
            
            with open(train_file, "w", encoding="utf-8") as f_train:
                f_train.writelines(lines[:n_train])
            with open(val_file, "w", encoding="utf-8") as f_val:
                f_val.writelines(lines[n_train:])
            print(f"    [SHUFFLE & SPLIT COMPLETE] Train: {n_train:,} lines ({train_pct}%) | Val: {n_val:,} lines ({val_pct}%)")

        # 4. Handle JSON Array
        elif file_path.name.endswith(".json"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            if isinstance(data, list):
                n_total = len(data)
                if shuffle and n_total > 1:
                    random.Random(seed).shuffle(data)
                n_val = max(1, int(n_total * val_ratio))
                n_train = n_total - n_val
                with open(train_file, "w", encoding="utf-8") as f_train:
                    json.dump(data[:n_train], f_train)
                with open(val_file, "w", encoding="utf-8") as f_val:
                    json.dump(data[n_train:], f_val)
                print(f"    [SHUFFLE & SPLIT COMPLETE] Train: {n_train:,} items ({train_pct}%) | Val: {n_val:,} items ({val_pct}%)")
            else:
                import shutil
                shutil.copy(file_path, train_file)
                shutil.copy(file_path, val_file)

        return train_file, val_file
    except Exception as e:
        print(f"    [ERROR] Failed to split {file_name}: {e}")
        if file_path.exists():
            print(f"    [CLEANUP] Removing corrupted raw file: {file_path}")
            file_path.unlink()
        return None, None


def download_stage(
    stage: str,
    shards_per_source: Optional[int] = 1,
    base_dir: str = "storage/dataset",
    split: bool = True,
    val_ratio: float = 0.05,
    seed: int = 42,
    shuffle: bool = True,
):
    """
    Download verified datasets for a specific stage into storage/dataset/{stage}/{domain}/
    and automatically apply structured shuffle and split into train/ and val/ subsets (95% Train / 5% Val).
    """
    if stage not in DATASET_REGISTRY:
        raise ValueError(f"Unknown stage '{stage}'. Valid stages are: {list(DATASET_REGISTRY.keys())}")

    stage_info = DATASET_REGISTRY[stage]
    base_path = Path(base_dir) / stage
    train_pct = int(round((1.0 - val_ratio) * 100))
    val_pct = int(round(val_ratio * 100))
    shard_desc = "ALL available" if (shards_per_source is None or shards_per_source <= 0) else f"up to {shards_per_source}"
    print(f"\n=================================================================")
    print(f"=== Downloading: {stage_info['title']} ===")
    print(f"=== Target Directory: {base_path.resolve()} ===")
    print(f"=== Shards: {shard_desc} per domain ===")
    print(f"=== Split: Train ({train_pct}%) / Val ({val_pct}%) | Shuffle: {'Enabled (seed=' + str(seed) + ')' if shuffle else 'Disabled'} ===")
    print(f"=================================================================")

    for domain_key, domain_info in stage_info["domains"].items():
        repo = domain_info["repo"]
        pattern = domain_info["pattern"]
        pct = int(domain_info["weight"] * 100)
        
        print(f"\n-> [{domain_info['name']}] (Weight: {pct}%)")
        print(f"   Fetching verified shards from: {repo} (pattern: {pattern})")

        max_shards = domain_info.get("max_production_shards", 5)
        if shards_per_source is not None and shards_per_source > 0:
            effective_limit = shards_per_source
        else:
            effective_limit = max_shards

        shards = get_verified_shards(repo, pattern=pattern, limit=effective_limit)
        if not shards:
            print(f"   [WARN] No matching shards found for {repo}")
            continue

        print(f"   Found {len(shards)} shard(s). Downloading...")
        domain_dir = base_path / domain_key
        raw_dir = domain_dir / "raw"
        train_dir = domain_dir / "train"
        val_dir = domain_dir / "val"

        for idx, rfilename in enumerate(shards, start=1):
            file_name = Path(rfilename).name
            raw_path = raw_dir / file_name
            success = download_shard(repo, rfilename, raw_path)
            
            if split and success and raw_path.exists():
                split_dataset_file(raw_path, train_dir, val_dir, val_ratio=val_ratio, seed=seed, shuffle=shuffle)


def download_corpus(
    stage: str = "all",
    shards_per_source: Optional[int] = 1,
    target_dir: str = "storage/dataset",
    split: bool = True,
    val_ratio: float = 0.05,
    seed: int = 42,
    shuffle: bool = True,
):
    """
    Dispatcher function to download dataset by stage or all stages with structured shuffle and train/val splitting.
    """
    if stage == "all":
        for s in DATASET_REGISTRY.keys():
            download_stage(s, shards_per_source=shards_per_source, base_dir=target_dir, split=split, val_ratio=val_ratio, seed=seed, shuffle=shuffle)
    else:
        download_stage(stage, shards_per_source=shards_per_source, base_dir=target_dir, split=split, val_ratio=val_ratio, seed=seed, shuffle=shuffle)

    print(f"\n[DONE] All requested datasets downloaded, structured-shuffled, and split into: {Path(target_dir).resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Mesosfer Bear AI: Multi-Stage Real Dataset Downloader & Structured Splitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m data.dataset --stage all --all-shards
  uv run python -m data.dataset --stage pretrain --all-shards
  uv run python -m data.dataset --stage cpt --all-shards
  uv run python -m data.dataset --stage sft --all-shards
  uv run python -m data.dataset --stage safety --all-shards
  uv run python -m data.dataset --stage all --shards 5
  uv run python -m data.dataset --info
        """
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["pretrain", "cpt", "sft", "safety", "all"],
        help="Training stage dataset to download (default: all)"
    )
    parser.add_argument(
        "--all-shards",
        action="store_true",
        help="Download ALL available production shards across all dataset repositories"
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=None,
        help="Number of shards to download per dataset source (default: all if --all-shards, else 1)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio (default: 0.05 = 95% train / 5% val)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for structured shuffling (default: 42)"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable deterministic dataset shuffling"
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Disable automatic train/validation splitting"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="storage/dataset",
        help="Root target directory to store downloaded datasets (default: storage/dataset)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Display dataset stage breakdown, proportions, and descriptions"
    )

    args = parser.parse_args()

    if args.info:
        print_stage_info()
        return

    # Determine shards limit
    if args.all_shards:
        shards_limit = 0
    elif args.shards is not None:
        shards_limit = args.shards
    else:
        shards_limit = 0  # Default to full production download

    train_pct = int(round((1.0 - args.val_ratio) * 100))
    val_pct = int(round(args.val_ratio * 100))
    print(f"=== Mesosfer Bear AI Dataset Downloader & Structured Splitter ===")
    print(f"Stage: {args.stage.upper()} | Shards Mode: {'ALL PRODUCTION SHARDS' if shards_limit == 0 else str(shards_limit) + ' per source'} | Directory: {args.target_dir}")
    print(f"Split Mode: {'Enabled (Train ' + str(train_pct) + '% / Val ' + str(val_pct) + '%)' if not args.no_split else 'Disabled'}")
    print(f"Shuffle   : {'Enabled (Seed ' + str(args.seed) + ')' if not args.no_shuffle else 'Disabled'}")

    download_corpus(
        stage=args.stage,
        shards_per_source=shards_limit,
        target_dir=args.target_dir,
        split=not args.no_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )


if __name__ == "__main__":
    main()

"""
Mesosfer Bear AI: Hugging Face Hub Dataset Exporter & Automated Uploader

This module packages and pushes the multi-stage, structured-shuffled (seed=42),
and 95/5 train/val split datasets from `storage/dataset/` to a Hugging Face Hub dataset repository.
Includes automatic Dataset Card (README.md) generation with domain breakdown and metadata tags.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from data.dataset import DATASET_REGISTRY


def load_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """
    Retrieve Hugging Face authentication token from CLI arg, environment variable, or .env file.
    """
    if cli_token:
        return cli_token.strip()

    # Check env
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()

    # Check .env file in workspace root
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
            if line.startswith("HUGGING_FACE_HUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    # Fallback to huggingface-cli cached token
    try:
        from huggingface_hub import get_token
        cached = get_token()
        if cached:
            return cached.strip()
    except Exception:
        pass

    return None


def generate_dataset_card_readme(repo_id: str, stage_filter: str = "all") -> str:
    """
    Generate a rich Hugging Face Dataset Card (README.md) with YAML frontmatter,
    formal stage breakdown tables, and quickstart code snippets.
    """
    readme = [
        "---",
        "license: apache-2.0",
        "task_categories:",
        "  - text-generation",
        "  - question-answering",
        "  - conversational",
        "language:",
        "  - id",
        "  - en",
        "tags:",
        "  - pretraining",
        "  - continual-pretraining",
        "  - sft",
        "  - safety-alignment",
        "  - code",
        "  - mathematics",
        "  - powershell",
        "  - mesosfer-bear",
        "pretty_name: Mesosfer Bear AI Multi-Stage Foundation Dataset",
        "size_categories:",
        "  - 10B<n<100B",
        "---",
        "",
        "# 🐻 Mesosfer Bear AI - Multi-Stage Foundation Corpus",
        "",
        "This repository contains the curated, structured-shuffled (`seed=42`), and 95% Train / 5% Validation split corpus used to pretrain and align **Mesosfer Bear AI** (a 16-layer transformer LLM optimized for high-efficiency training on AMD Instinct MI300X).",
        "",
        "## 📊 Dataset Structure & Stage Breakdown",
        "",
    ]

    for stage_key, stage_info in DATASET_REGISTRY.items():
        if stage_filter != "all" and stage_filter != stage_key:
            continue

        readme.append(f"### {stage_info['title']}")
        readme.append("| Domain | Proposer / Source Repo | Weight | Target Format | Focus & Scope |")
        readme.append("| :--- | :--- | :--- | :--- | :--- |")

        for dom_key, dom in stage_info["domains"].items():
            pct = f"{int(dom['weight'] * 100)}%"
            readme.append(f"| `{dom_key}` | `{dom['repo']}` | **{pct}** | Parquet / JSONL | {dom['description']} |")
        readme.append("")

    readme.extend([
        "## 🛠️ Data Preprocessing & Splitting Protocol",
        "- **Structured Deterministic Shuffling**: Every individual shard is shuffled with `seed=42` using zero-copy PyArrow array indexing to prevent gradient correlation and eliminate loss spikes.",
        "- **95% Train / 5% Validation Split**: Every single domain shard is partitioned into `train/` and `val/` directories.",
        "- **Sequence Length Target**: Built for `max_seq_len = 4096` tokens with `<|endoftext|>` sequence packing.",
        "",
        "## 🚀 Quickstart: Loading Data in Python",
        "```python",
        "from datasets import load_dataset",
        "",
        f"# Load Pretrain Bahasa Indonesia Train Split",
        f"dataset = load_dataset('{repo_id}', data_dir='pretrain/bahasa_umum_id/train')",
        "print(dataset)",
        "```",
        "",
        "## 📜 Licensing & Attribution",
        "All constituent datasets belong to their respective original authors and maintainers. Distributed under Apache-2.0 in compliance with upstream open-source licenses.",
    ])

    return "\n".join(readme)


def scan_local_dataset_files(base_dir: Path, stage: str = "all") -> Dict[str, List[Path]]:
    """
    Scan local directory structure and return dictionary of files grouped by stage.
    """
    files_by_stage: Dict[str, List[Path]] = {}
    
    stages_to_scan = [stage] if stage != "all" else list(DATASET_REGISTRY.keys())
    
    for s in stages_to_scan:
        stage_dir = base_dir / s
        if not stage_dir.exists():
            continue
        
        file_list: List[Path] = []
        for file_path in stage_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.endswith(".tmp"):
                file_list.append(file_path)
        files_by_stage[s] = file_list
        
    return files_by_stage


def upload_dataset_to_hf(
    repo_id: str,
    source_dir: str = "storage/dataset",
    stage: str = "all",
    token: Optional[str] = None,
    private: bool = True,
    dry_run: bool = False,
) -> bool:
    """
    Upload local storage/dataset files to Hugging Face Hub dataset repository.
    """
    from huggingface_hub import HfApi

    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"[ERROR] Source directory does not exist: {source_path.resolve()}")
        return False

    auth_token = load_hf_token(token)
    if not auth_token and not dry_run:
        print("[ERROR] No Hugging Face token found. Provide --token or set HF_TOKEN in .env")
        return False

    api = HfApi(token=auth_token)

    # 1. Scan local files
    files_by_stage = scan_local_dataset_files(source_path, stage=stage)
    total_files = sum(len(files) for files in files_by_stage.values())
    total_bytes = sum(f.stat().st_size for files in files_by_stage.values() for f in files)

    print("=" * 70)
    print(f"=== MESOSFER BEAR AI - HUGGING FACE DATASET UPLOADER ===")
    print(f"Target Repository : {repo_id}")
    print(f"Target Stage      : {stage.upper()}")
    print(f"Local Source Dir  : {source_path.resolve()}")
    print(f"Total Files       : {total_files} file(s)")
    print(f"Total Size        : {total_bytes / (1024 * 1024 * 1024):.2f} GB ({total_bytes / (1024 * 1024):.1f} MB)")
    print(f"Visibility        : {'Private' if private else 'Public'}")
    print(f"Dry-Run Mode      : {'ENABLED' if dry_run else 'DISABLED'}")
    print("=" * 70)

    for s, files in files_by_stage.items():
        s_bytes = sum(f.stat().st_size for f in files)
        print(f"  - [{s.upper()}]: {len(files)} files | {s_bytes / (1024 * 1024):.1f} MB")

    if total_files == 0:
        print("\n[WARN] No dataset files found to upload. Please make sure dataset is downloaded first.")
        return False

    if dry_run:
        print("\n[DRY RUN] Verification complete. No changes pushed to Hugging Face.")
        return True

    # 2. Ensure Remote Dataset Repo Exists
    try:
        print(f"\n-> Checking / Creating Hugging Face dataset repository: {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"   [OK] Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"   [ERROR] Failed to create / access repo {repo_id}: {e}")
        return False

    # 3. Write Dataset Card (README.md) locally so it is committed in the SAME single commit
    try:
        readme_content = generate_dataset_card_readme(repo_id=repo_id, stage_filter=stage)
        local_readme_path = (source_path / "README.md") if stage == "all" else (source_path / stage / "README.md")
        local_readme_path.write_text(readme_content, encoding="utf-8")
        print(f"   [OK] Local Dataset Card (README.md) prepared for atomic commit.")
    except Exception as e:
        print(f"   [WARN] Could not write local README.md: {e}")

    # 4. Upload Whole Folder in ONE SINGLE ATOMIC COMMIT (zero rate-limit risk)
    try:
        commit_msg = (
            f"feat(dataset): upload Bear AI complete multi-stage corpus"
            if stage == "all"
            else f"feat(dataset): upload stage {stage} Bear AI corpus"
        )

        upload_dir = source_path if stage == "all" else (source_path / stage)
        path_in_repo = None if stage == "all" else stage

        print(f"-> Uploading all files in ONE SINGLE ATOMIC COMMIT from {upload_dir.resolve()} to {repo_id}...")
        api.upload_folder(
            folder_path=str(upload_dir.resolve()),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_msg,
        )

        print(f"\n[SUCCESS] Dataset uploaded in 1 atomic commit (Zero rate-limit overhead)!")
        print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
        return True

    except Exception as e:
        print(f"\n[ERROR] Dataset upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Mesosfer Bear AI: Hugging Face Dataset Exporter & Uploader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m data.hf_upload --repo Dummy9898/bear-dataset --stage all
  uv run python -m data.hf_upload --repo Dummy9898/bear-dataset --stage pretrain
  uv run python -m data.hf_upload --repo Dummy9898/bear-dataset --dry-run
  uv run python -m data.hf_upload --repo Dummy9898/bear-dataset --public
        """
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="Dummy9898/bear-dataset",
        help="Target Hugging Face Hub repository ID (default: Dummy9898/bear-dataset)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["pretrain", "cpt", "sft", "safety", "all"],
        help="Stage to upload (default: all)"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="storage/dataset",
        help="Local directory path containing datasets (default: storage/dataset)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face write token (optional, auto-read from .env / HF_TOKEN)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the Hugging Face dataset public (default: private)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate upload and inspect file list/sizes without pushing to Hugging Face"
    )

    args = parser.parse_args()

    success = upload_dataset_to_hf(
        repo_id=args.repo,
        source_dir=args.source_dir,
        stage=args.stage,
        token=args.token,
        private=not args.public,
        dry_run=args.dry_run,
    )

    if not success and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()

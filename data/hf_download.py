"""
Mesosfer Bear AI: Hugging Face Dataset Downloader & Restorer

Downloads the pre-processed, deterministic-shuffled (seed=42), and train/val partitioned
multi-stage dataset directly from the Hugging Face Hub dataset repository into `storage/dataset/`.

Usage:
  uv run python -m data.hf_download --repo Dummy9898/bear-dataset --stage all
  uv run python -m data.hf_download --repo Dummy9898/bear-dataset --stage pretrain
  uv run python -m data.hf_download --repo Dummy9898/bear-dataset --dry-run
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict

from data.hf_upload import load_hf_token


def download_dataset_from_hf(
    repo_id: str = "Dummy9898/bear-dataset",
    dest_dir: str = "storage/dataset",
    stage: str = "all",
    token: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """
    Download dataset shards from Hugging Face Hub repository into local destination folder.
    """
    from huggingface_hub import HfApi, snapshot_download

    auth_token = load_hf_token(token)
    api = HfApi(token=auth_token)

    dest_path = Path(dest_dir)

    print("=" * 70)
    print("=== MESOSFER BEAR AI - HUGGING FACE DATASET DOWNLOADER ===")
    print(f"Source Repository : {repo_id}")
    print(f"Target Stage      : {stage.upper()}")
    print(f"Destination Dir   : {dest_path.resolve()}")
    print(f"Dry-Run Mode      : {'ENABLED' if dry_run else 'DISABLED'}")
    print("=" * 70)

    # 1. Inspect Remote Repository
    try:
        print(f"\n-> Checking remote dataset repository '{repo_id}'...")
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        # Filter files by stage
        if stage != "all":
            matched_files = [f for f in repo_files if f.startswith(f"{stage}/")]
        else:
            matched_files = [f for f in repo_files if any(f.startswith(s + "/") for s in ["pretrain", "cpt", "sft", "safety"])]

        print(f"   [OK] Found {len(matched_files)} dataset file(s) on remote repository.")

    except Exception as e:
        print(f"   [ERROR] Could not access Hugging Face repository '{repo_id}': {e}")
        print("   Make sure your HF_TOKEN is configured in .env or passed via --token.")
        return False

    if dry_run:
        print("\n[DRY RUN] Remote dataset shard list preview:")
        for f in matched_files[:10]:
            print(f"   - {f}")
        if len(matched_files) > 10:
            print(f"   ... and {len(matched_files) - 10} more files.")
        print(f"\nTotal remote files to download: {len(matched_files)}")
        return True

    # 2. Download via snapshot_download with parallel acceleration
    try:
        dest_path.mkdir(parents=True, exist_ok=True)
        allow_patterns = [f"{stage}/**"] if stage != "all" else ["pretrain/**", "cpt/**", "sft/**", "safety/**", "README.md"]

        print(f"\n-> Downloading dataset files into {dest_path.resolve()}...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest_path.resolve()),
            allow_patterns=allow_patterns,
            token=auth_token,
            max_workers=8,
        )

        # Count downloaded files
        local_files = [f for f in dest_path.rglob("*") if f.is_file() and not f.name.endswith(".tmp")]
        total_size = sum(f.stat().st_size for f in local_files)

        print(f"\n[SUCCESS] Dataset successfully downloaded and restored!")
        print(f"Local Path  : {dest_path.resolve()}")
        print(f"Total Files : {len(local_files)} files")
        print(f"Total Size  : {total_size / (1024 * 1024 * 1024):.2f} GB ({total_size / (1024 * 1024):.1f} MB)")
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Mesosfer Bear AI: Hugging Face Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m data.hf_download --repo Dummy9898/bear-dataset --stage all
  uv run python -m data.hf_download --repo Dummy9898/bear-dataset --stage pretrain
  uv run python -m data.hf_download --repo Dummy9898/bear-dataset --dry-run
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
        help="Stage to download (default: all)"
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="storage/dataset",
        help="Destination directory to store downloaded dataset (default: storage/dataset)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face read/write token (optional, auto-read from .env / HF_TOKEN)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check remote repository files and verify connection without downloading"
    )

    args = parser.parse_args()

    success = download_dataset_from_hf(
        repo_id=args.repo,
        dest_dir=args.dest_dir,
        stage=args.stage,
        token=args.token,
        dry_run=args.dry_run,
    )

    if not success and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()

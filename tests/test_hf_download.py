"""Tests for data.hf_download — Hugging Face dataset downloading and dry-run inspection."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from data.hf_download import download_dataset_from_hf


def test_download_dataset_dry_run():
    with patch("huggingface_hub.HfApi.list_repo_files") as mock_list:
        mock_list.return_value = [
            "pretrain/bahasa_umum_id/train/shard_000.parquet",
            "cpt/matematika/train/shard_000.parquet",
            "sft/percakapan_id/train/shard_000.parquet",
            "safety/safety_pku/train/shard_000.parquet",
            "README.md"
        ]
        success = download_dataset_from_hf(
            repo_id="Dummy9898/bear-dataset",
            dest_dir="storage/dataset",
            stage="all",
            dry_run=True,
        )
        assert success is True


def test_download_dataset_stage_filter():
    with patch("huggingface_hub.HfApi.list_repo_files") as mock_list:
        mock_list.return_value = [
            "pretrain/bahasa_umum_id/train/shard_000.parquet",
            "cpt/matematika/train/shard_000.parquet",
        ]
        success = download_dataset_from_hf(
            repo_id="Dummy9898/bear-dataset",
            dest_dir="storage/dataset",
            stage="pretrain",
            dry_run=True,
        )
        assert success is True

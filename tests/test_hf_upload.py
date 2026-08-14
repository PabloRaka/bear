"""
Unit Tests for Mesosfer Bear AI Hugging Face Dataset Uploader
"""

import pytest
from pathlib import Path
from data.hf_upload import (
    load_hf_token,
    generate_dataset_card_readme,
    scan_local_dataset_files,
    upload_dataset_to_hf,
)


def test_load_hf_token_from_cli():
    token = load_hf_token("hf_test_cli_token_12345")
    assert token == "hf_test_cli_token_12345"


def test_load_hf_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env_token_abcdef")
    token = load_hf_token(None)
    assert token == "hf_env_token_abcdef"


def test_generate_dataset_card_readme_contains_all_stages():
    readme = generate_dataset_card_readme(repo_id="test-org/bear-test-corpus", stage_filter="all")
    
    assert "---" in readme
    assert "license: apache-2.0" in readme
    assert "Mesosfer Bear AI" in readme
    assert "Stage 1: Pre-training" in readme
    assert "Stage 2: Continued Pre-Training" in readme
    assert "Stage 3: Supervised Fine-Tuning" in readme
    assert "Stage 4: Safety & Guardrails" in readme
    assert "test-org/bear-test-corpus" in readme
    assert "seed=42" in readme
    assert "95% Train / 5% Validation" in readme


def test_generate_dataset_card_readme_filtered_stage():
    readme = generate_dataset_card_readme(repo_id="test-org/bear-test-corpus", stage_filter="sft")
    assert "Stage 3: Supervised Fine-Tuning" in readme
    assert "Stage 1: Pre-training" not in readme


def test_scan_local_dataset_files_filters_tmp(tmp_path):
    pretrain_dir = tmp_path / "pretrain" / "bahasa_id"
    pretrain_dir.mkdir(parents=True)
    
    valid_file = pretrain_dir / "data.parquet"
    valid_file.write_text("dummy parquet content")
    
    tmp_file = pretrain_dir / "download.tmp"
    tmp_file.write_text("partial download")
    
    scanned = scan_local_dataset_files(tmp_path, stage="all")
    assert "pretrain" in scanned
    assert valid_file in scanned["pretrain"]
    assert tmp_file not in scanned["pretrain"]


def test_upload_dataset_to_hf_dry_run(tmp_path):
    stage_dir = tmp_path / "pretrain" / "test_domain"
    stage_dir.mkdir(parents=True)
    sample_file = stage_dir / "train.parquet"
    sample_file.write_text("dummy data")
    
    success = upload_dataset_to_hf(
        repo_id="dummy/test-repo",
        source_dir=str(tmp_path),
        stage="pretrain",
        token="hf_mock_token",
        private=True,
        dry_run=True,
    )
    assert success is True

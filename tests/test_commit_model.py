"""Tests for scripts.commit_model — Packaging and bundling model checkpoints."""

import json
import pytest
from pathlib import Path
import torch

from engine.transformer import BearConfig, BearTransformer
from scripts.commit_model import package_and_export_bundle, find_available_checkpoints


def test_package_and_export_bundle(tmp_path: Path):
    # 1. Create a dummy model checkpoint
    config = BearConfig(n_layers=2, d_model=128, ffn_hidden=256, n_heads=4, n_kv_heads=2)
    model = BearTransformer(config)
    
    ckpt_path = tmp_path / "dummy_checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(),
        "step": 12000,
        "loss": 1.42,
        "model_config": config,
    }, ckpt_path)
    
    export_dir = tmp_path / "export_test"
    success, params = package_and_export_bundle(
        checkpoint_path=ckpt_path,
        stage="pretrain",
        output_dir=export_dir,
        repo_id="test-org/bear-test",
    )
    
    assert success is True
    assert (export_dir / "bear_model.pt").exists()
    assert (export_dir / "config.json").exists()
    assert (export_dir / "generation_config.json").exists()
    assert (export_dir / "inference.py").exists()
    assert (export_dir / "README.md").exists()
    assert (export_dir / "engine" / "transformer.py").exists()
    
    with open(export_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg["n_layers"] == 2

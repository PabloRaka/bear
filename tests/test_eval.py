"""Tests for eval.evaluator — stage evaluation benchmarks and metrics."""

import pytest
import torch
from pathlib import Path

from engine.transformer import BearTransformer, BearConfig
from engine.tokenizer import BearTokenizer
from eval.evaluator import (
    load_task_file,
    compute_sequence_loss_and_bpb,
    generate_text,
    eval_pretrain_stage,
    eval_cpt_stage,
    eval_sft_stage,
    eval_safety_stage,
    run_stage_evaluation,
)


@pytest.fixture
def mock_tokenizer():
    tok_path = Path("storage/tokenizer/bear_tokenizer.json")
    if tok_path.exists():
        return BearTokenizer.load(str(tok_path))
    return BearTokenizer()


@pytest.fixture
def mock_model(mock_tokenizer):
    cfg = BearConfig(
        vocab_size=mock_tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=128,
        max_seq_len=512,
    )
    return BearTransformer(cfg)


def test_load_all_task_files():
    for stage in ["pretrain", "cpt", "sft", "safety"]:
        data = load_task_file(stage, task_dir="task")
        assert "stage" in data
        assert data["stage"] == stage
        assert "tasks" in data
        assert len(data["tasks"]) > 0
        assert "huggingface_sources" in data


def test_compute_sequence_loss_and_bpb(mock_model, mock_tokenizer):
    loss, ppl, bpb = compute_sequence_loss_and_bpb(
        model=mock_model,
        tokenizer=mock_tokenizer,
        prompt="Indonesia adalah negara",
        target=" kepulauan di Asia.",
        device="cpu",
    )
    assert loss > 0
    assert ppl >= 1.0
    assert bpb > 0


def test_generate_text(mock_model, mock_tokenizer):
    output = generate_text(
        model=mock_model,
        tokenizer=mock_tokenizer,
        prompt="Halo Bear AI",
        max_new_tokens=10,
        device="cpu",
    )
    assert isinstance(output, str)


def test_eval_pretrain_stage(mock_model, mock_tokenizer):
    tasks_data = load_task_file("pretrain", task_dir="task")
    res = eval_pretrain_stage(mock_model, mock_tokenizer, tasks_data, device="cpu")
    assert "summary" in res
    assert res["summary"]["avg_loss"] > 0
    assert res["summary"]["avg_bpb"] > 0
    assert len(res["details"]) == len(tasks_data["tasks"])


def test_eval_cpt_stage(mock_model, mock_tokenizer):
    tasks_data = load_task_file("cpt", task_dir="task")
    res = eval_cpt_stage(mock_model, mock_tokenizer, tasks_data, device="cpu")
    assert "summary" in res
    assert "pattern_match_rate" in res["summary"]


def test_eval_sft_stage(mock_model, mock_tokenizer):
    tasks_data = load_task_file("sft", task_dir="task")
    res = eval_sft_stage(mock_model, mock_tokenizer, tasks_data, device="cpu")
    assert "summary" in res
    assert "avg_keyword_recall" in res["summary"]


def test_eval_safety_stage(mock_model, mock_tokenizer):
    tasks_data = load_task_file("safety", task_dir="task")
    res = eval_safety_stage(mock_model, mock_tokenizer, tasks_data, device="cpu")
    assert "summary" in res
    assert "refusal_accuracy" in res["summary"]


def test_run_stage_evaluation_dry_run(tmp_path):
    report = run_stage_evaluation(
        stage="all",
        tokenizer_path="storage/tokenizer/bear_tokenizer.json",
        task_dir="task",
        device="cpu",
        dry_run=True,
    )
    assert "stages" in report
    assert "pretrain" in report["stages"]
    assert "cpt" in report["stages"]
    assert "sft" in report["stages"]
    assert "safety" in report["stages"]

"""Tests for chat.cli — streaming generation and chat formatting."""

import pytest
import torch
from pathlib import Path

from engine.transformer import BearTransformer, BearConfig
from engine.tokenizer import BearTokenizer
from chat.cli import stream_generate, DEFAULT_SYSTEM_PROMPT


@pytest.fixture
def tokenizer():
    tok_path = Path("storage/tokenizer/bear_tokenizer.json")
    if tok_path.exists():
        return BearTokenizer.load(str(tok_path))
    return BearTokenizer()


@pytest.fixture
def model(tokenizer):
    cfg = BearConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=128,
        max_seq_len=256,
    )
    return BearTransformer(cfg)


def test_chat_template_rendering(tokenizer):
    convo = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "Halo Bear AI!"}
    ]
    prompt_text = tokenizer.apply_chat_template(convo, thinking=True, tokenize=False)
    assert "<|open|>message role=\"system\"<|close|>" in prompt_text
    assert "<|open|>message role=\"user\"<|close|>" in prompt_text
    assert "thinking=\"max\"" in prompt_text
    assert "<|end_of_msg|>" in prompt_text

    prompt_ids = tokenizer.apply_chat_template(convo, thinking=True, tokenize=True)
    assert isinstance(prompt_ids, list)
    assert len(prompt_ids) > 0


def test_stream_generate(model, tokenizer):
    convo = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ]
    prompt_ids = tokenizer.apply_chat_template(convo, thinking=False, tokenize=True)
    response, count, speed = stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        max_new_tokens=16,
        temperature=0.7,
        device="cpu",
    )
    assert isinstance(response, str)
    assert count > 0
    assert speed > 0

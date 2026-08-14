import pytest
import os
from engine.tokenizer import BearTokenizer


@pytest.fixture
def tokenizer():
    path = "storage/tokenizer/bear_tokenizer.json"
    assert os.path.exists(path), f"Tokenizer artifact not found at {path}"
    return BearTokenizer.from_file(path)


def test_vocab_size_is_60k(tokenizer):
    assert tokenizer.vocab_size == 60000, f"Expected vocab size 60000, got {tokenizer.vocab_size}"


def test_special_tokens_within_bounds(tokenizer):
    assert len(tokenizer.special_tokens) == 14
    for tok_name, tok_id in tokenizer.special_tokens.items():
        assert 0 <= tok_id < 60000, f"Special token {tok_name} ID {tok_id} out of bounds [0, 60000)"


def test_lossless_roundtrip_multi_domain(tokenizer):
    samples = [
        "Halo dunia! Model Mesosfer Bear AI siap untuk melayani berbagai domain data secara akurat.",
        "The quick brown fox jumps over the lazy dog. Artificial intelligence is evolving rapidly.",
        "def compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:\n    return torch.matmul(q, k.transpose(-2, -1))",
        "Get-ChildItem -Path .\\storage\\ -Recurse | Where-Object { $_.Length -gt 1MB } | Select-Object FullName",
        "sudo systemctl restart nginx && tail -f /var/log/nginx/error.log | grep --color=auto '502'",
        "git checkout -b feature/tokenizer-60k && git commit -m 'feat: 60k vocab tokenizer'",
        "docker run -d --name bear-db -p 5432:5432 -e POSTGRES_PASSWORD=secret postgres:16-alpine",
        r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}, \quad \nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
        "Step 1: Identify boundary conditions. Step 2: Formulate matrix Laplacian. Step 3: Solve linear system.",
    ]

    for text in samples:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Lossless roundtrip failed for:\nOriginal: {text}\nDecoded: {decoded}"


def test_xtml_chat_template(tokenizer):
    messages = [
        {"role": "system", "content": "You are Bear AI assistant."},
        {"role": "user", "content": "Berikan contoh script PowerShell untuk filter file."},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert "<|open|>message role=\"system\"<|close|>" in prompt
    assert "<|open|>message role=\"assistant\" thinking=\"max\"<|close|>" in prompt

    tokens = tokenizer.encode(prompt, allow_special=True)
    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    assert decoded == prompt

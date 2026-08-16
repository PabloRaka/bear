"""
Mesosfer Bear AI - Interactive Chat CLI

Interactive terminal chat interface for Mesosfer Bear AI models.
Supports:
  - Kimi K3 / XTML multi-turn chat template formatting.
  - Token-by-token real-time streaming response generation.
  - Interactive commands: /clear, /system, /thinking, /temp, /history, /help, /exit.
  - Automatic device dispatch (CUDA / ROCm / Apple Silicon MPS / CPU).

Usage:
  uv run python -m chat.cli --checkpoint storage/models/bear_final.pt
  uv run python -m chat.cli --checkpoint storage/models/bear_step_5000.pt --temperature 0.7
  uv run python -m chat.cli --dry-run
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from engine.transformer import BearTransformer, BearConfig
from engine.tokenizer import BearTokenizer


DEFAULT_SYSTEM_PROMPT = "Anda adalah Mesosfer Bear AI, asisten kecerdasan buatan yang cerdas, sopan, dan berbahasa Indonesia baku serta fasih dalam pemrograman dan sains."


def print_banner(model_name: str, params: int, vocab_size: int, device: str, thinking: bool):
    """Print clean and informative CLI banner."""
    print("=" * 68)
    print("   __  __                             ____                  _    ___ ")
    print("  |  \\/  | ___  ___  ___  ___  __ _|  _ \\  ___  __ _ _ __  / \\  |_ _|")
    print("  | |\\/| |/ _ \\/ __|/ _ \\/ __|/ _` | |_) |/ _ \\/ _` | '__|/ _ \\  | | ")
    print("  | |  | |  __/\\__ \\ (_) \\__ \\ (_| |  _ <|  __/ (_| | |  / ___ \\ | | ")
    print("  |_|  |_|\\___||___/\\___/|___/\\__,_|_| \\_\\\\___|\\__,_|_| /_/   \\_\\___|")
    print("=" * 68)
    print(f"  Model       : {model_name} ({params/1e6:.1f}M parameters)")
    print(f"  Tokenizer   : BearTokenizer (Vocab: {vocab_size:,})")
    print(f"  Device      : {device}")
    print(f"  XTML Format : Kimi-K3 Compliant | Thinking: {'ON' if thinking else 'OFF'}")
    print("=" * 68)
    print("  Perintah: /help, /clear, /system <prompt>, /thinking <on|off>, /exit")
    print("=" * 68 + "\n")


def stream_generate(
    model: BearTransformer,
    tokenizer: BearTokenizer,
    prompt_ids: List[int],
    max_new_tokens: int = 512,
    min_new_tokens: int = 1,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    device: str = "cpu",
    stop_token_ids: Optional[set] = None,
) -> Tuple[str, int, float]:
    """
    Stream token-by-token generation to stdout in real-time.
    Returns (generated_text, num_tokens_generated, tokens_per_sec).
    """
    model.eval()
    if stop_token_ids is None:
        stop_token_ids = {
            tokenizer.special_tokens.get("<|end_of_msg|>"),
            tokenizer.special_tokens.get("<|end_of_text|>"),
            tokenizer.special_tokens.get("[EOT]"),
            tokenizer.eos_id,
        }
        stop_token_ids = {tid for tid in stop_token_ids if tid is not None}

    # Crop to max context window
    max_ctx = model.config.max_seq_len - max_new_tokens
    if max_ctx > 0 and len(prompt_ids) > max_ctx:
        prompt_ids = prompt_ids[-max_ctx:]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids: List[int] = []

    t0 = time.time()
    for step_i in range(max_new_tokens):
        cond_ids = input_ids if input_ids.size(1) <= model.config.max_seq_len else input_ids[:, -model.config.max_seq_len:]
        with torch.no_grad():
            logits, _ = model(cond_ids)
            next_logits = logits[:, -1, :].clone()

            # Repetition penalty
            if generated_ids:
                for prev_token in set(generated_ids[-128:]):
                    if next_logits[0, prev_token] < 0:
                        next_logits[0, prev_token] *= 1.15
                    else:
                        next_logits[0, prev_token] /= 1.15

            if temperature > 0:
                next_logits = next_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[:, indices_to_remove] = float("-inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        if step_i >= min_new_tokens and token_id in stop_token_ids:
            break

        generated_ids.append(token_id)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stream token piece to console
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        sys.stdout.write(token_str)
        sys.stdout.flush()

    elapsed = max(time.time() - t0, 1e-4)
    tokens_per_sec = len(generated_ids) / elapsed
    full_response = tokenizer.decode(generated_ids, skip_special_tokens=False)

    return full_response, len(generated_ids), tokens_per_sec


def run_chat_loop(
    model: BearTransformer,
    tokenizer: BearTokenizer,
    device: str = "cpu",
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    thinking: bool = True,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    """Main interactive REPL chat session."""
    conversation: List[Dict[str, str]] = []

    print_banner(
        model_name="Mesosfer Bear",
        params=model.param_count(),
        vocab_size=tokenizer.vocab_size,
        device=device.upper(),
        thinking=thinking,
    )

    while True:
        try:
            user_input = input("\n\033[1;36mUser >\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSesi chat diakhiri. Sampai jumpa!")
            break

        if not user_input:
            continue

        # CLI Commands
        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

            if cmd in ("/exit", "/quit", "/q"):
                print("Sesi chat diakhiri. Sampai jumpa!")
                break
            elif cmd in ("/clear", "/c"):
                conversation.clear()
                print("\033[1;33m[Riwayat percakapan berhasil dibersihkan]\033[0m")
                continue
            elif cmd == "/system":
                if arg:
                    system_prompt = arg
                    print(f"\033[1;33m[System prompt diperbarui]:\033[0m {system_prompt}")
                else:
                    print(f"\033[1;33m[System prompt saat ini]:\033[0m {system_prompt}")
                continue
            elif cmd == "/thinking":
                if arg.lower() in ("on", "true", "1", "yes"):
                    thinking = True
                    print("\033[1;33m[Thinking Mode]: ON\033[0m")
                elif arg.lower() in ("off", "false", "0", "no"):
                    thinking = False
                    print("\033[1;33m[Thinking Mode]: OFF\033[0m")
                else:
                    print(f"\033[1;33m[Thinking Mode]:\033[0m {'ON' if thinking else 'OFF'}")
                continue
            elif cmd == "/temp":
                try:
                    temperature = float(arg)
                    print(f"\033[1;33m[Temperature diatur ke]:\033[0m {temperature}")
                except ValueError:
                    print(f"\033[1;31m[Error]: Masukkan angka valid. Contoh: /temp 0.7\033[0m")
                continue
            elif cmd == "/history":
                print(f"\n--- Riwayat Percakapan ({len(conversation)} pesan) ---")
                for msg in conversation:
                    print(f"[{msg['role'].upper()}]: {msg['content']}")
                print("--------------------------------------------------")
                continue
            elif cmd == "/help":
                print("\n\033[1;32m=== Perintah Chat CLI ===\033[0m")
                print("  /clear             : Menghapus riwayat percakapan")
                print("  /system <teks>     : Mengubah instruksi system prompt")
                print("  /thinking <on|off> : Mengaktifkan/menonaktifkan tag thinking XTML")
                print("  /temp <nilai>      : Mengatur temperature sampling (cth: 0.7)")
                print("  /history           : Melihat riwayat percakapan")
                print("  /exit, /quit       : Keluar dari program")
                continue
            else:
                print(f"\033[1;31mPerintah tidak dikenal '{cmd}'. Ketik /help untuk bantuan.\033[0m")
                continue

        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})

        # Build full conversation message list with system prompt
        full_convo = [{"role": "system", "content": system_prompt}] + conversation

        # Apply Kimi-K3 XTML chat markup template
        prompt_ids = tokenizer.apply_chat_template(
            full_convo,
            add_generation_prompt=True,
            thinking=thinking,
            tokenize=True,
        )

        sys.stdout.write("\n\033[1;32mBear AI >\033[0m ")
        sys.stdout.flush()

        response, token_count, tok_per_sec = stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )

        sys.stdout.write(f"\n\033[2m[{token_count} tokens | {tok_per_sec:.1f} tok/s]\033[0m\n")
        sys.stdout.flush()

        # Save assistant response to conversation
        conversation.append({"role": "assistant", "content": response.strip()})


def main():
    parser = argparse.ArgumentParser(description="Mesosfer Bear AI - Interactive Chat CLI")
    parser.add_argument("--checkpoint", type=str, default="storage/models/bear_final.pt", help="Path to trained checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="storage/tokenizer/bear_tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering threshold")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) filtering threshold")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum generated tokens per response")
    parser.add_argument("--thinking", action="store_true", default=True, help="Enable XTML thinking format")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="Initial system prompt")
    parser.add_argument("--dry-run", action="store_true", help="Launch instantly with lightweight in-memory model")

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # 1. Load Tokenizer
    tok_path = Path(args.tokenizer)
    if tok_path.exists():
        tokenizer = BearTokenizer.load(str(tok_path))
    else:
        print(f"[Warning] Tokenizer not found at {tok_path}, using base fallback.")
        tokenizer = BearTokenizer()

    # 2. Load Model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists() and not args.dry_run:
        for cand in (
            "storage/models/safety/bear_final.pt",
            "storage/models/sft/bear_final.pt",
            "storage/models/cpt/bear_final.pt",
            "storage/models/pretrain/bear_final.pt",
            "storage/models/bear_final.pt",
        ):
            if Path(cand).exists():
                ckpt_path = Path(cand)
                break

    if args.dry_run or not ckpt_path.exists():
        if not args.dry_run:
            print(f"[Notice] Checkpoint {ckpt_path} not found. Running in mock/dry-run mode.")
        cfg = BearConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            ffn_hidden=256,
            max_seq_len=1024,
        )
        model = BearTransformer(cfg).to(device)
    else:
        print(f"Loading model checkpoint from: {ckpt_path} ...")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model_cfg = BearConfig(**ckpt.get("model_config", {}))
        model = BearTransformer(model_cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Checkpoint successfully loaded! Model parameters: {model.param_count():,}")

    # 3. Start Chat REPL
    run_chat_loop(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
        thinking=args.thinking,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()

from typing import List, Optional

import torch
import torch.nn as nn

from src.transformer_core.model import MiniTransformerLM
from src.data_pipeline.tokenizer_utils import load_tokenizer


def load_model_and_tokenizer(
    ckpt_path: str = "models/mini_lm/checkpoint.pt",
    tokenizer_path: str = "data/tokenizer/tokenizer.json",
    device: str = "cuda",
) -> tuple[MiniTransformerLM, object]:
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    model = MiniTransformerLM(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=4,
        ff_dim=1024,
        num_layers=4,
        max_seq_len=128,
        dropout=0.0,
    )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(
    model: MiniTransformerLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: str = "cuda",
) -> str:
    """
    Basit autoregressive text generation (greedy/top-k sampling).
    Şimdilik KV cache yok; her adımda tüm sekansı yeniden geçiriyoruz.
    """
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    # Prompt'u tokenize et
    input_ids = tokenizer.encode(prompt).ids
    input_ids = [bos_id] + input_ids
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq)

    for _ in range(max_new_tokens):
        # Eğer seq_len > max_seq_len ise son max_seq_len token'ı al
        if input_ids.size(1) > model.max_seq_len:
            input_ids = input_ids[:, -model.max_seq_len :]

        logits = model(input_ids)  # (1, seq, vocab)
        logits = logits[:, -1, :]  # son token'ın logits'i

        # Temperature
        logits = logits / max(temperature, 1e-6)

        if top_k is not None:
            # top-k dışını -inf yap
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        next_token_id = next_id.item()
        # EOS gelirse dur
        if next_token_id == eos_id:
            break

        input_ids = torch.cat([input_ids, next_id], dim=1)

    # BOS'u at, EOS'a kadar decode et
    out_ids = input_ids[0].tolist()
    if eos_id in out_ids:
        out_ids = out_ids[1 : out_ids.index(eos_id)]
    else:
        out_ids = out_ids[1:]

    text = tokenizer.decode(out_ids)
    return text

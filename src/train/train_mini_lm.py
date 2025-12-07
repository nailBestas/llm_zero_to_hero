import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.transformer_core.model import MiniTransformerLM
from src.data_pipeline.dataset import TextLineDataset
from src.data_pipeline.tokenizer_utils import load_tokenizer


def create_dataloader(
    files: List[str],
    tokenizer_path: str,
    max_seq_len: int = 128,
    batch_size: int = 32,
    num_workers: int = 2,
) -> DataLoader:
    tokenizer = load_tokenizer(tokenizer_path)
    dataset = TextLineDataset(files=files, tokenizer=tokenizer, max_seq_len=max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, tokenizer


def create_model(vocab_size: int, device: str = "cuda") -> MiniTransformerLM:
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=4,
        ff_dim=1024,
        num_layers=4,
        max_seq_len=128,
        dropout=0.1,
    )
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str = "cuda",
    log_interval: int = 100,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(input_ids)
            # CrossEntropyLoss: (batch*seq, vocab) vs (batch*seq)
            vocab_size = logits.size(-1)
            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tokens = labels.numel()
        total_loss += loss.item() * tokens
        total_tokens += tokens

        if (step + 1) % log_interval == 0:
            avg_loss = total_loss / total_tokens
            ppl = math.exp(avg_loss)
            print(f"Step {step+1}: loss={avg_loss:.4f}, ppl={ppl:.2f}")

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Basit bir config; istersen sonra config/ dosyalarına taşıyabilirsin
    data_files = [str(p) for p in Path("data/raw").glob("*.txt")]
    tokenizer_path = "data/tokenizer/tokenizer.json"
    batch_size = 16
    max_seq_len = 128
    lr = 3e-4
    num_epochs = 1
    ckpt_path = "models/mini_lm/checkpoint.pt"

    assert len(data_files) > 0, "data/raw içinde en az bir .txt dosyan olmalı."

    dataloader, tokenizer = create_dataloader(
        files=data_files,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=2,
    )

    vocab_size = tokenizer.get_vocab_size()
    model = create_model(vocab_size=vocab_size, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            log_interval=50,
        )
        ppl = math.exp(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: avg_loss={avg_loss:.4f}, ppl={ppl:.2f}")
        save_checkpoint(model, optimizer, ckpt_path)


if __name__ == "__main__":
    main()

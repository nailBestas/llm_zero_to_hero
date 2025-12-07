import math
import os
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
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
) -> (DataLoader, object):
    tokenizer = load_tokenizer(tokenizer_path)
    dataset = TextLineDataset(files=files, tokenizer=tokenizer, max_seq_len=max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, tokenizer


def create_model(vocab_size: int, device: torch.device) -> MiniTransformerLM:
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
    scaler: torch.amp.GradScaler,
    device: torch.device,
    log_interval: int = 100,
    rank: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(input_ids)
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

        if rank == 0 and (step + 1) % log_interval == 0:
            avg_loss = total_loss / total_tokens
            ppl = math.exp(avg_loss)
            print(f"[rank {rank}] Step {step+1}: loss={avg_loss:.4f}, ppl={ppl:.2f}")

    # Tüm rank'ler arasında loss toplama (isteğe bağlı, basitlik için sadece local hesaplıyoruz)
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


def setup_distributed(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


def ddp_worker(rank: int, world_size: int):
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    data_files = [str(p) for p in Path("data/raw").glob("*.txt")]
    tokenizer_path = "data/tokenizer/tokenizer.json"
    batch_size = 16
    max_seq_len = 128
    lr = 3e-4
    num_epochs = 1
    ckpt_dir = Path("models/mini_lm_ddp")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_rank{rank}.pt"

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

    ddp_model = DDP(model, device_ids=[rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            model=ddp_model,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            log_interval=50,
            rank=rank,
        )
        if rank == 0:
            ppl = math.exp(avg_loss)
            print(f"[rank {rank}] Epoch {epoch+1}/{num_epochs}: avg_loss={avg_loss:.4f}, ppl={ppl:.2f}")

    # Her rank kendi checkpoint'ini kaydedebilir (veya sadece rank 0)
    if rank == 0:
        torch.save(
            {
                "model_state_dict": ddp_model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"[rank {rank}] Saved checkpoint to {ckpt_path}")

    cleanup_distributed()


def main():
    world_size = torch.cuda.device_count()
    assert world_size >= 1, "En az 1 GPU gerekiyor; 2+ GPU ile daha anlamlıdır."
    print("World size (GPUs):", world_size)

    mp.spawn(ddp_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

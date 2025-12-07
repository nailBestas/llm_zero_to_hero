from pathlib import Path

import torch

from src.data_pipeline.tokenizer_utils import train_bpe_tokenizer, load_tokenizer
from src.data_pipeline.dataset import TextLineDataset


def test_tokenizer_train_and_load(tmp_path: Path):
    # Geçici küçük bir text dosyası
    text_path = tmp_path / "tiny.txt"
    text_path.write_text("hello world\nthis is a test\n", encoding="utf-8")

    tok_path = tmp_path / "tokenizer.json"
    tokenizer = train_bpe_tokenizer(
        files=[str(text_path)],
        vocab_size=100,
        min_frequency=1,
        save_path=str(tok_path),
    )

    assert tok_path.exists()
    tok2 = load_tokenizer(str(tok_path))
    ids = tok2.encode("hello world").ids
    assert isinstance(ids, list)
    assert len(ids) > 0


def test_text_line_dataset(tmp_path: Path):
    # Küçük text + tokenizer
    text_path = tmp_path / "tiny.txt"
    text_path.write_text("hello world\nthis is a test\n", encoding="utf-8")

    tok_path = tmp_path / "tokenizer.json"
    tokenizer = train_bpe_tokenizer(
        files=[str(text_path)],
        vocab_size=100,
        min_frequency=1,
        save_path=str(tok_path),
    )

    ds = TextLineDataset(files=[str(text_path)], tokenizer=tokenizer, max_seq_len=16)
    it = iter(ds)
    sample = next(it)

    assert "input_ids" in sample and "labels" in sample
    assert isinstance(sample["input_ids"], torch.Tensor)
    assert isinstance(sample["labels"], torch.Tensor)
    assert sample["input_ids"].shape == sample["labels"].shape
    assert sample["input_ids"].shape[0] >= 2

from pathlib import Path
from typing import List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_bpe_tokenizer(
    files: List[str],
    vocab_size: int = 32000,
    min_frequency: int = 2,
    save_path: str = "data/tokenizer/tokenizer.json",
) -> Tokenizer:
    """
    Basit bir BPE tokenizer eğitir ve kaydeder.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )

    tokenizer.train(files=files, trainer=trainer)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    return tokenizer


def load_tokenizer(path: str = "data/tokenizer/tokenizer.json") -> Tokenizer:
    """
    Eğitilmiş tokenizer'ı yükler.
    """
    return Tokenizer.from_file(path)

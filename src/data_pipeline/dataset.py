from pathlib import Path
from typing import Iterator, List, Dict

import torch
from torch.utils.data import IterableDataset
from tokenizers import Tokenizer

from .tokenizer_utils import load_tokenizer


class TextLineDataset(IterableDataset):
    """
    Büyük text dosyalarını satır satır okuyup token ID'lerine çeviren basit streaming dataset.
    """

    def __init__(
        self,
        files: List[str],
        tokenizer: Tokenizer = None,
        max_seq_len: int = 256,
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
    ):
        super().__init__()
        self.files = [str(f) for f in files]
        self.tokenizer = tokenizer or load_tokenizer()
        self.max_seq_len = max_seq_len
        self.bos_id = self.tokenizer.token_to_id(bos_token)
        self.eos_id = self.tokenizer.token_to_id(eos_token)

    def _line_iterator(self) -> Iterator[str]:
        for fp in self.files:
            path = Path(fp)
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    def _encode_line(self, line: str) -> torch.Tensor:
        ids = self.tokenizer.encode(line).ids
        # BOS + ids + EOS, sonra max_seq_len'e göre kırp
        full_ids = [self.bos_id] + ids + [self.eos_id]
        full_ids = full_ids[: self.max_seq_len]
        return torch.tensor(full_ids, dtype=torch.long)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for line in self._line_iterator():
            input_ids = self._encode_line(line)
            # Basit bir next-token hedefi: shift by one
            if len(input_ids) < 2:
                continue
            x = input_ids[:-1]
            y = input_ids[1:]
            yield {
                "input_ids": x,
                "labels": y,
            }

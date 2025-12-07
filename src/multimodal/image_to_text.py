from typing import Optional

import torch

from src.multimodal.vision_encoder import VisionEncoder
from src.inference.generation import load_model_and_tokenizer, generate


class ImageToTextPipeline:
    """
    Basit image -> text pipeline:
    1) CLIP ile image embedding
    2) Embedding bilgisini prompt'a göm
    3) Mini LLM ile text üret
    """

    def __init__(
        self,
        ckpt_path: str = "models/mini_lm/checkpoint.pt",
        tokenizer_path: str = "data/tokenizer/tokenizer.json",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vision = VisionEncoder(device=self.device)
        self.model, self.tokenizer = load_model_and_tokenizer(
            ckpt_path=ckpt_path, tokenizer_path=tokenizer_path, device=self.device
        )

    def describe_image(
        self,
        image_path: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> str:
        # 1) Görsel embedding al (şimdilik sadece var/yok bilgisi için kullanıyoruz)
        _embeds = self.vision.encode_image(image_path)

        # 2) Basit bir prompt şablonu (modelin dili neyse ona göre uyarlayabilirsin)
        prompt = "Resmi kısaca açıkla:"

        # 3) LLM ile açıklama üret
        text = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None,
            device=self.device,
        )
        return text

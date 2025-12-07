from typing import Tuple

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class VisionEncoder:
    """
    Basit CLIP tabanlı görsel encoder.
    image -> CLIP image embedding (tensor)
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.inference_mode()
    def encode_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)  # (1, dim)
        # Normalize (isteğe bağlı)
        embeds = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return embeds  # (1, dim)

from pathlib import Path

from PIL import Image
import torch

from src.multimodal.vision_encoder import VisionEncoder


def test_vision_encoder_runs(tmp_path: Path):
    # Küçük boş bir resim oluştur
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    encoder = VisionEncoder(device="cpu")
    embeds = encoder.encode_image(str(img_path))
    assert isinstance(embeds, torch.Tensor)
    assert embeds.ndim == 2
    assert embeds.shape[0] == 1

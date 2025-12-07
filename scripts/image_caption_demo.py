import argparse
import torch

from src.multimodal.image_to_text import ImageToTextPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Resim dosyası yolu")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    pipe = ImageToTextPipeline(device=device)
    out = pipe.describe_image(
        image_path=args.image,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print("Caption:", out)


if __name__ == "__main__":
    main()

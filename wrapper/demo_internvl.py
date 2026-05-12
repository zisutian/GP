import argparse
from pathlib import Path

from internvl_wrapper import DEFAULT_MODEL_PATH, InternVLWrapper


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "InternVL" / "internvl_chat" / "examples" / "image1.jpg"


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal InternVL2.5-1B wrapper demo.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--question", default="Please describe the image shortly.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-dynamic-patch", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    wrapper = InternVLWrapper(
        model_path=args.model_path,
        device=args.device,
        max_dynamic_patch=args.max_dynamic_patch,
    )
    response = wrapper.chat(
        question=args.question,
        image_path=args.image,
        max_new_tokens=args.max_new_tokens,
    )
    print(response)


if __name__ == "__main__":
    main()

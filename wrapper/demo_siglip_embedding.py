import argparse
from pathlib import Path

from siglip_embedding_wrapper import (
    DEFAULT_INTERNVL_PATH,
    DEFAULT_PALIGEMMA_PATH,
    DEFAULT_SIGLIP_PATH,
    SigLIPPrefixInternVLWrapper,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "InternVL" / "internvl_chat" / "examples" / "image1.jpg"


def parse_args():
    parser = argparse.ArgumentParser(description="Inject SigLIP image embeddings into InternVL's language model.")
    parser.add_argument("--internvl-path", default=str(DEFAULT_INTERNVL_PATH))
    parser.add_argument("--siglip-path", default=DEFAULT_SIGLIP_PATH)
    parser.add_argument("--paligemma-path", default=str(DEFAULT_PALIGEMMA_PATH))
    parser.add_argument("--projector-path", default=None)
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--question", default="Please describe the image shortly.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--train-language-model", action="store_true", help="Unfreeze the LLM; off by default.")
    return parser.parse_args()


def main():
    args = parse_args()
    wrapper = SigLIPPrefixInternVLWrapper(
        internvl_path=args.internvl_path,
        siglip_path=args.siglip_path,
        paligemma_path=args.paligemma_path,
        projector_path=args.projector_path,
        device=args.device,
        train_language_model=args.train_language_model,
    )
    response = wrapper.generate(
        question=args.question,
        image=args.image,
        max_new_tokens=args.max_new_tokens,
    )
    print(response)


if __name__ == "__main__":
    main()

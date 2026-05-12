# conda create -n 260513-internvl python=3.9 -y
# conda activate 260513-internvl

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

cd "$(dirname "${BASH_SOURCE[0]}")/InternVL"
pip install -r requirements/internvl_chat.txt

pip install pandas lmdb opencv-python tensorboardX

# Optional but recommended: keep pretrained weights inside this project so
# training scripts can run offline/reproducibly after the first download.
mkdir -p pretrained/OpenGVLab

# Download the assembled InternVL2.5-1B checkpoint used by 2nd finetune.
# Run this inside the 260513-internvl environment when network is available.
huggingface-cli download OpenGVLab/InternVL2_5-1B \
  --local-dir pretrained/OpenGVLab/InternVL2_5-1B \
  --local-dir-use-symlinks False

# If you later run stage1 from separate vision/LLM components, download these too.
# huggingface-cli download OpenGVLab/InternViT-300M-448px-V2_5 \
#   --local-dir pretrained/InternViT-300M-448px-V2_5 \
#   --local-dir-use-symlinks False
# huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
#   --local-dir pretrained/Qwen2.5-0.5B-Instruct \
#   --local-dir-use-symlinks False

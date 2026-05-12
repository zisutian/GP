# conda create -n 260513-internvl python=3.9 -y
# conda activate 260513-internvl

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

cd "$(dirname "${BASH_SOURCE[0]}")/InternVL"
pip install -r requirements/internvl_chat.txt

pip install pandas lmdb opencv-python tensorboardX

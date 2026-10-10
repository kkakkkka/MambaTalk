
# 🐍 MambaTalk

> **Official PyTorch implementation of [MambaTalk: Efficient Holistic Gesture Synthesis with Selective State Space Models](https://arxiv.org/pdf/2403.09471).**

MambaTalk leverages the power of Selective State Space Models to achieve efficient and high-quality holistic gesture synthesis.

## 📝 Release Plans

* [x] Inference codes and pretrained weights
* [x] Training scripts

## 🛠️ Installation

### Environment Setup

We recommend **Python 3.9.21** and **CUDA 12.2**. Please follow the steps below to set up the environment:

```shell
git clone https://github.com/kkakkkka/MambaTalk -b main
cd MambaTalk

# [Optional] Create a virtual env
conda create -n mambatalk python==3.9.21
conda activate mambatalk

# 1. Install system dependencies
# ffmpeg for media processing and libstdcxx-ng for rendering
conda install -c conda-forge libstdcxx-ng ffmpeg

# 2. Install PyTorch and basic requirements
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Install PyTorch3D
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'

# 4. Install Mamba and Causal Conv1d (Pre-compiled wheels for Linux x86_64)
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.1cxx11abiTRUE-cp39-cp39-linux_x86_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

```

### 📥 Download Weights

You can download the pretrained weights from [Huggingface](https://huggingface.co/xuzn/MambaTalk/tree/main) and place them in the `./pretrained/` directory:

```shell
pip3 install "huggingface_hub[cli]"
huggingface-cli download --resume-download thuteam/MambaTalk --local-dir pretrained

```

**Directory Structure:**
Ensure your `pretrained` folder is organized as follows:

```text
./pretrained/
├── pretrained_vq
│   ├── face.bin
│   ├── foot.bin
│   ├── hands.bin
│   ├── lower_foot.bin
│   └── upper.bin
├── smplx_models
│   └── smplx
│       └── SMPLX_NEUTRAL_2020.npz
├── test_sequences
└── mambatalk_100.bin

```

## 🚀 Training and Inference

### 1. Data Preparation

Download and unzip the **BEAT2** dataset via Hugging Face to your root directory:

```shell
git lfs install
git clone https://huggingface.co/datasets/H-Liu1997/BEAT2

```

### 2. Evaluation of Pretrained Weights

Once the BEAT2 dataset is ready, you can run the evaluation script:

```shell
bash run_scripts/test.sh

```

### 3. Customized Data Processing

If you wish to use your own data, please organize it in the following structure:

```text
.
├── smplxflame_30
│   ├── 2_scott_0_1_1.npz
│   └── 2_scott_0_2_2.npz
├── test.csv
├── textgrid
│   ├── 2_scott_0_1_1.TextGrid
│   └── 2_scott_0_2_2.TextGrid
└── wave16k
    ├── 2_scott_0_1_1.wav
    └── 2_scott_0_2_2.wav

```

**Format of `test.csv`:**

```csv
id,type
2_scott_0_1_1,test
2_scott_0_2_2,test

```

**Audio Alignment (TextGrid Generation):**
We recommend using **Montreal Forced Aligner (MFA)** to generate `TextGrid` files from speech recordings.

```shell
# Install MFA and dependencies
pip install git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
conda install -c conda-forge kalpy
pip install pgvector Bio

# Download models
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Align data (Place speech recordings in ./data first)
mfa align ./data english_us_arpa english_us_arpa ./data/result

```

### 4. 🎥 Visualize Generated Results

After generating the `.npz` files, visualize the output using the rendering script:

```shell
npy_path="./res_2_scott_0_1_1.npz"
wav_path="./BEAT2/beat_english_v2.0.0/wave16k/2_scott_0_1_1.wav"
save_dir="outputs/render"

xvfb-run -a python render.py --npy_path $npy_path --wav_path $wav_path --save_dir $save_dir

```

---

## 🏋️ Training

### Train MambaTalk (Main Model)

```shell
bash run_scripts/train.sh

```

### Train VQ-VAEs (Components)

You can train individual VQ-VAE components using the following commands:

```shell
# Face
python train.py --config ./configs/cnn_vqvae_face_30.yaml 

# Hands
python train.py --config configs/cnn_vqvae_hands_30.yaml 

# Lower Body
python train.py --config configs/cnn_vqvae_lower_30.yaml 

# Lower Foot
python train.py --config configs/cnn_vqvae_lower_foot_30.yaml 

# Upper Body
python train.py --config configs/cnn_vqvae_upper_30.yaml 

```

## 📖 Citation

If you find MambaTalk useful for your research, please consider citing:

```bibtex
@article{xu2024mambatalk,
  title={Mambatalk: Efficient holistic gesture synthesis with selective state space models},
  author={Xu, Zunnan and Lin, Yukang and Han, Haonan and Yang, Sicheng and Li, Ronghui and Zhang, Yachao and Li, Xiu},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={20055--20080},
  year={2024}
}

```

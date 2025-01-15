# MambaTalk

## 📝 Release Plans

- [X] Inference codes and pretrained weights
- [X] Training scripts

## ⚒️ Installation

### Build Environtment

We Recommend a python version `==3.9.21` and cuda version `==12.2`. Then build environment as follows:

```shell
git clone https://github.com/kkakkkka/MambaTalk -b main
# [Optional] Create a virtual env
conda create -n mambatalk python==3.9.21
conda activate mambatalk
# Install ffmpeg for media processing and libstdcxx-ng for rendering
conda install -c conda-forge libstdcxx-ng ffmpeg
# Install with pip:
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.1cxx11abiTRUE-cp39-cp39-linux_x86_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

### Download weights

You may run the following command to download weights in ``./pretrained/``:

```shell
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/xuzn/MambaTalk pretrained
```

These weights should be orgnized as follows:

```text
./pretrained/
|-- pretrained_vq
|   |-- face.bin
|   |-- foot.bin
|   |-- hands.bin
|   |-- lower_foot.bin
|   |-- upper.bin
|-- smplx_models
|   |-- smplx/SMPLX_NEUTRAL_2020.npz
|-- test_sequences
|-- mambatalk_100.bin
```

## 🚀 Training and Inference

### Data Preparation

Download the unzip version BEAT2 via hugging face in path ``<your root>``:

```shell
git lfs install
git clone https://huggingface.co/datasets/H-Liu1997/BEAT2
```

### Evaluation of Pretrained Weights

After you downloaded BEAT2 dataset, run:

```shell
bash run_scripts/test.sh
```

### Training of MambaTalk

```shell
bash run_scripts/train.sh
```

### Training of VQVAEs

```shell
python train.py --config ./configs/cnn_vqvae_face_30.yaml 
```

```shell
python train.py --config configs/cnn_vqvae_hands_30.yaml 
```

```shell
python train.py --config configs/cnn_vqvae_lower_30.yaml 
```

```shell
python train.py --config configs/cnn_vqvae_lower_foot_30.yaml 
```

```shell
python train.py --config configs/cnn_vqvae_upper_30.yaml 
```

## Acknowledgements

The code is based on [EMAGE](https://github.com/PantoMatrix/PantoMatrix). We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

## Citation

If MambaTalk is useful for your research, please consider citing:

```angular2html
@article{xu2024mambatalk,
  title={Mambatalk: Efficient holistic gesture synthesis with selective state space models},
  author={Xu, Zunnan and Lin, Yukang and Han, Haonan and Yang, Sicheng and Li, Ronghui and Zhang, Yachao and Li, Xiu},
  journal={arXiv preprint arXiv:2403.09471},
  year={2024}
}
```

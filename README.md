# MambaTalk

This is an official PyTorch implementation of [MambaTalk: Efficient Holistic Gesture Synthesis with Selective State Space Models](https://arxiv.org/pdf/2403.09471).

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

You may run the following command to download weights from [Huggingface](https://huggingface.co/xuzn/MambaTalk/tree/main) in ``./pretrained/``:

```shell
pip3 install "huggingface_hub[cli]"
huggingface-cli download --resume-download kkakkkka/MambaTalk --local-dir pretrained
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

### Customized Data

For your own data, you should organize it as follows:

```shell
.
├── smplxflame_30
│   ├── 2_scott_0_1_1.npz
│   ├── 2_scott_0_2_2.npz
├── test.csv
├── textgrid
│   ├── 2_scott_0_1_1.TextGrid
│   ├── 2_scott_0_2_2.TextGrid
├── wave16k
│   ├── 2_scott_0_1_1.wav
│   ├── 2_scott_0_2_2.wav
```

In `test.csv`, please list your files as shown below:

```shell
id,type
2_scott_0_1_1,test
2_scott_0_2_2,test
```

If you want to generate corresponding TextGrid files from your speech recordings, we recommend installing Montreal Forced Aligner (MFA). These aligned text files should then be used as input alongside your audio files.

```bash
pip install git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
conda install -c conda-forge kalpy
pip install pgvector
pip install Bio
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
# Please put your speech recordings under ./data first
mfa align ./data english_us_arpa english_us_arpa ./data/result
```

### Visualize the Generated Results

With generated npy files, you can visualize the results using command below:

```shell
npy_path="./res_2_scott_0_1_1.npz"
wav_path="./BEAT2/beat_english_v2.0.0/wave16k/2_scott_0_1_1.wav"
save_dir="outputs/render"

xvfb-run -a python render.py --npy_path $npy_path --wav_path $wav_path --save_dir $save_dir
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
@inproceedings{xu2024mambatalk,
  title={Mambatalk: Efficient holistic gesture synthesis with selective state space models},
  author={Xu, Zunnan and Lin, Yukang and Han, Haonan and Yang, Sicheng and Li, Ronghui and Zhang, Yachao and Li, Xiu},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

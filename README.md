# Skin-UMamba: An efficient and lightweight method for skin lesion segmentation 🐍

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
This repo provides an efficient and lightweight method for skin lesion dermoscopy images. Skin-UMamba is a hybrid CNN-Mamba-KAN architecture with key advantages: (i) efficiently capturing long-range dependencies with linear complexity, and (ii) learning highly nonlinear lesion boundaries using only 0.690M parameters and 0.180 GFLOPs, significantly reducing segmentation errors and providing clinicians with more accurate morphological analysis of lesions. Extensive experiments on the ISIC 2017, ISIC 2018, and PH2 datasets demonstrate that Skin-UMamba achieves superior performance and generalization capabilities. 

---

## ⚙️ 1. Installation
### Main environment (Python 3.8)
```bash
conda create -n skumamba python=3.8 -y
conda activate skumamba

# PyTorch 1.13 + CUDA 11.7
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# dependencies
pip install packaging timm==0.4.12 pytest chardet yacs termcolor submitit tensorboardX triton==2.0.0

# Mamba series (wheels built)
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1

# other libs
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy
```
### Data preprocessing tool (Python 3.7)
```bash
conda create -n dataprocess python=3.7 -y
conda activate dataprocess
pip install h5py
conda install scipy=1.2.1
pip install pillow
```
---

## 📁 2. Dataset Preparation

| **Dataset**    | **Folder**                  | **Script**                                           |
| ---------- | ----------------------- | ------------------------------------------------ |
| ISIC 2017  | `/data/dataset_isic17/` | `python dataprepare/Prepare_ISIC2017.py`         |
| ISIC 2018  | `/data/dataset_isic18/` | `python dataprepare/Prepare_ISIC2018.py`         |
| PH2      | `/data/PH2/`            | `python dataprepare/Prepare_PH2.py`              |
| Custom| `/data/your_dataset/`   | edit & run `dataprepare/Prepare_your_dataset.py` |

**All datasets can be downloaded following the instructions in [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet/tree/main).**

---

## 🏋️ 3. Train
```bash
conda activate vmunet
python train.py
```
**Results saved to ./results/**

---

## 🧪 4. Test
**Edit resume_model path in test.py**
```bash
python test.py
```
**After testing, you could obtain the outputs in './results/'**

🤝 Acknowledgement
Thanks to Vim, VMamba, VM-UNet and LightM-UNet for their excellent work.

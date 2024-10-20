# LightForker: LightFormer for 2024

The goal of this repository is to understand and rewrite the LightFormer model proposed by Ming et al. 2023.

# Install Requirements

1. Clone this repository.
2. Make sure you have Python3.8 installed, checking using `python3.8 --version`
3. Set up a virtual environment using `python3.8 -m venv .venv` and `source .venv/bin/activate`
4. Update pip using `pip install --upgrade pip`
5. Install dependencies: `pip install -r requirements.txt`
6. Done!

# Datasets

LightFormer2 is implemented for

- LISA Traffic Light Dataset: https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset

If desired, can also use:

- Bosch Small Traffic Lights Dataset: https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset

## Basic Dataset Setup

1. Install and unzip datasets from links above.
2. Copy `frames` folder directly into their respective places.
3. Specify paths to each directory containing a frames folder and a samples.json label file in the `train.py` and `test.py` code.

# Training

In src directory:

PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train.py

# Testing

In src directory:

PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train.py

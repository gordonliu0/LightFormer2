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

LightFormer uses two datasets:

- Bosch Small Traffic Lights Dataset: https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset
- LISA Traffic Light Dataset: https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset

## Basic Dataset Setup

1. Install datasets from links above.
2. Unzip.
3. Copy `frames` folder into their respective places.

## Custom Dataset Setup

1. (Optional) If you'd like to set up a custom structure, you can update sample_database_folder in `configs/generate_config.py`. Keep in mind each directory listed must have a `.json` for labels and a `frames` for images.
2. Run `python3 configs/generate_config.py` to generate configs.
3. Note that if you are keeping this on github make sure to .gitignore the image directories!

# Training

In src directory:

PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train.py

# Testing

In src directory:

PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train.py

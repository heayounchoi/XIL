# XIL: Cross-Expanding Incremental Learning

Official implementation of  
**[XIL: Cross-Expanding Incremental Learning](https://openreview.net/pdf?id=eaAGI1lIb4)**  
ICLR 2026

This repository contains the cleaned core implementation of the XIL protocol and the XEED model.

The current code supports:
- `centroids`: select representative class/domain samples with ViT feature clustering.
- `train`: train task prompts/classifiers and build prototype classifiers from centroid/prototype data.
- `eval`: run domain-key selection and prompt-conditioned inference.

## Project Structure

```text
XEED/
├── configs/
├── data.py
├── model.py
├── trainer.py
├── metrics.py
├── utils.py
├── run.py
├── requirements.txt
└── README.md
```

## Installation

Create and activate a conda environment.

```bash
conda create -n xeed python=3.8 -y
conda activate xeed
```

Install PyTorch according to your CUDA version. For CUDA 11.8:

```bash
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies.

```bash
pip install -r requirements.txt
```

## Data Preparation

The code expects dataset split files in the following format:

```text
relative/path/to/image.jpg class_id
relative/path/to/image2.jpg class_id
```

For each domain, prepare train/test files such as:

```text
clipart_train.txt
clipart_test.txt
painting_train.txt
painting_test.txt
real_train.txt
real_test.txt
sketch_train.txt
sketch_test.txt
```

The exact domain names and order should match the `domain_order` field in the config file.


## Running XEED

XEED is run in three stages:

1. Extract centroids.
2. Train task prompts/classifiers and build prototype classifiers.
3. Evaluate with domain-key selection and prompt-conditioned inference.

## 1. Extract Centroids

This step selects representative samples for each class/domain using ViT feature clustering.

```bash
python run.py \
  --config configs/domainnet.json \
  --run_mode centroids \
  --centroid_output_dir ../data/domainnet/centroids
```

The selected centroid samples are saved under `centroid_output_dir`.

## 2. Train

This step trains task-specific prompts and classifiers. It also builds prototype classifiers from centroid/prototype data.

```bash
python run.py \
  --config configs/domainnet.json \
  --run_mode train \
  --proto_data_path ../data/domainnet/centroids \
  --output_dir logs
```

Checkpoints and logs are saved under `output_dir`.

## 3. Evaluate

This step loads a trained checkpoint and performs inference using domain-key selection, task/domain-specific prompt conditioning, and prototype-based classifier weights.

```bash
python run.py \
  --config configs/domainnet.json \
  --run_mode eval \
  --checkpoint logs/checkpoint.pth
```

## Notes

- XEED model-related code is based on the S-Prompts repository:  
  https://github.com/iamwangyabin/S-Prompts

- The current repository was refactored from the original XIL/XIL codebase with AI assistance. Some errors may remain after refactoring. Please report any issues if you find them.

- The data generation code for XEED 'representation modulation with domain semantics' component will be uploaded later.

- The generation process was based on InstantStyle:  
  https://github.com/instantX-research/InstantStyle

## Acknowledgement

This codebase builds on S-Prompts and InstantStyle.

Please refer to the original repositories for their implementation details and licenses:

- S-Prompts: https://github.com/iamwangyabin/S-Prompts
- InstantStyle: https://github.com/instantX-research/InstantStyle
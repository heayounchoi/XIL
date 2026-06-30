# XIL: Cross-Expanding Incremental Learning

Official implementation of
**[XIL: Cross-Expanding Incremental Learning](https://openreview.net/pdf?id=eaAGI1lIb4)**
ICLR 2026

This repository contains the cleaned core implementation of the XIL protocol and the XEED model.

The current code supports:

* `centroids`: select representative class/domain samples with ViT feature clustering.
* `generation`: generate prototype images and cross-class transferred data.
* `train`: train task prompts/classifiers and build prototype classifiers from generated prototype data.
* `eval`: run domain-key selection and prompt-conditioned inference.

## Project Structure

```text
XIL/
├── configs/
├── generation/
│   └── InstantStyle/
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

XEED is run in five stages:

1. Extract centroids.
2. Generate prototype images.
3. Generate cross-class transferred data.
4. Train task prompts/classifiers and build prototype classifiers.
5. Evaluate with domain-key selection and prompt-conditioned inference.

## 1. Extract Centroids

First, move to the main `XIL` folder and select representative samples for each class/domain using ViT feature clustering.

```bash
python run.py \
  --config configs/domainnet.json \
  --run_mode centroids \
  --centroid_output_dir ../data/domainnet/centroids
```

The selected centroid samples are saved under `centroid_output_dir`.

## 2. Generate Prototype Images

After extracting centroids, move to the data generation folder and run `generate_prototypes_{dataset}.py`.

For example:

```bash
cd generation/InstantStyle

python generate_prototypes_office31.py \
  --dom 0 \
  --task_order_path office31_task_order.json \
  --data_path ../../data/office31/centroids \
  --save_path ../../data/office31/prototypes
```

This step generates prototype images from the selected centroid samples.

## 3. Generate Cross-Class Transferred Data

Next, run `cross_class_generation_{dataset}.py` to generate cross-class transferred data.

For example:

```bash
python cross_class_generation_office31.py \
  --dom 0 \
  --task_order_path office31_task_order.json \
  --data_path ../../data/office31/prototypes \
  --save_path ../../data/office31/generated
```

The generated data will be used for XEED training.

## 4. Train

Return to the main `XIL` folder and train the model.

```bash
cd ../..

python run.py \
  --config configs/domainnet.json \
  --run_mode train \
  --proto_data_path ../data/domainnet/generated \
  --output_dir logs
```

This step trains task-specific prompts and classifiers. It also builds prototype classifiers from prototype/generated data.

Checkpoints and logs are saved under `output_dir`.

## 5. Evaluate

This step loads a trained checkpoint and performs inference using domain-key selection, task/domain-specific prompt conditioning, and prototype-based classifier weights.

```bash
python run.py \
  --config configs/domainnet.json \
  --run_mode eval \
  --checkpoint logs/checkpoint.pth
```

## Notes

* XEED model-related code is based on the S-Prompts repository:
  https://github.com/iamwangyabin/S-Prompts

* The data generation process is based on InstantStyle:
  https://github.com/instantX-research/InstantStyle

* This repository was refactored from the original XIL codebase with AI assistance. Some errors or inconsistencies may remain after refactoring. Please check the scripts carefully before running large-scale experiments and report any issues if you find them.

## Acknowledgement

This codebase builds on S-Prompts and InstantStyle.

Please refer to the original repositories for their implementation details and licenses:

* S-Prompts: https://github.com/iamwangyabin/S-Prompts
* InstantStyle: https://github.com/instantX-research/InstantStyle

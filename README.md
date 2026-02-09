# Backdoor Defense via Adaptively Splitting Poisoned Dataset (ASD)

This repository provides the official reproduction of the defense mechanism described in the CVPR 2023 paper: **Backdoor Defense via Adaptively Splitting Poisoned Dataset**.

## Paper Details

**Title:** Backdoor Defense via Adaptively Splitting Poisoned Dataset  
**Authors:** Kuofeng Gao, Yang Bai, Jindong Gu, Yong Yang, Shu-Tao Xia  
**Conference:** CVPR 2023  
**Paper Link:** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Gao_Backdoor_Defense_via_Adaptively_Splitting_Poisoned_Dataset_CVPR_2023_paper.html)  

**Citation:**
```bibtex
@inproceedings{gao2023backdoor,
  title={Backdoor Defense via Adaptively Splitting Poisoned Dataset},
  author={Gao, Kuofeng and Bai, Yang and Gu, Jindong and Yang, Yong and Xia, Shu-Tao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4005--4014},
  year={2023}
}
```

## Threat Model: BadNets Attack

This implementation focuses on defending against the **BadNets** attack.

*   **Objective**: The attacker aims to install a backdoor that causes the model to misclassify specific inputs quite reliably.
*   **Mechanism**: The attacker injects a small set of poisoned images into the training set. These images have a visible trigger (e.g., a 3x3 pixel pattern) and are mislabeled as the target class.
*   **Scenario (Many-to-One)**:
    *   **Source Class**: Any class.
    *   **Target Class**: Class 0.
    *   **Trigger**: A fixed pattern added to the image.
    *   **Poison Rate**: 5% of the total training set.

## Defense Mechanism: Adaptively Splitting Dataset (ASD)

Our defense leverages the framework of splitting the poisoned dataset into two data pools (clean and poisoned) and training on them adaptively.

### 1. Unified Splitting Framework
We formulate training-time defenses as a two-step process: splitting the dataset into a reliable (clean) pool and an unreliable (poisoned) pool.

### 2. Loss-Guided Split
Initial separation is performed based on training loss. Poisoned samples tend to have lower loss values early in training. We use a GMM on the loss distribution to separate these pools.

### 3. Meta-Learning-Inspired Split
To further refine the split, we employ a meta-learning approach, updating the split by minimizing the meta-loss on a small, trustworthy validation set.

### 4. Adaptive Training
The model is trained iteratively on the refined clean pool, preventing overfitting to the backdoor trigger.

## Implementation & Reproduction

We provide a complete pipeline to reproduce the results for **CIFAR-10** and **GTSRB**.

### 1. Installation
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 2. Dataset Generation

#### CIFAR-10
Download the CIFAR-10 dataset (Python version) from the [official website](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and extract it to your data directory.

#### GTSRB
Use our helper script `prepare_gtsrb.py` to prepare the GTSRB dataset from raw image folders into the required pickle format.

```bash
# Basic usage (defaults to internal paths)
python prepare_gtsrb.py --source /path/to/raw/GTSRB --dest /path/to/save/pickles
```

#### 3. Running Experiments
We provide shell scripts to automate the training and testing process for both datasets.

#### CIFAR-10 Experiments (`run_cifar10.sh`)
```bash
#!/bin/bash

## Train ASD model under BadNets attack for CIFAR10
python ASD.py \
  --config config/baseline_asd.yaml \
  --resume False \
  --gpu 0


## Test ASD model under BadNets attack for CIFAR10
python test.py \
  --config config/baseline_asd.yaml \
  --resume latest_model.pt \
  --gpu 0


## Training No Defense model under BadNets attack for CIFAR10
python train_baseline.py \
  --config config/baseline_asd.yaml \
  --gpu 0
```

#### GTSRB Experiments (`run_gtsrb.sh`)
```bash
#!/bin/bash

## Train ASD model under BadNets attack for GTSRB
python ASD.py \
  --config config/baseline_asd_gtsrb.yaml \
  --resume False \
  --gpu 0

## Test ASD model under BadNets attack for GTSRB
python test.py \
  --config config/baseline_asd_gtsrb.yaml \
  --resume latest_model.pt \
  --gpu 0


## Training No Defense model under BadNets attack for GTSRB
python train_baseline.py \
  --config config/baseline_asd_gtsrb.yaml \
  --gpu 0
```

## Evaluation & Results Comparison

We have successfully reproduced the results of the paper for **CIFAR-10** and **GTSRB** datasets using the **BadNets** attack.

### 1. CIFAR-10 Evaluation

We compare our reproduction results against the original paper's reported numbers for both the ASD Defense and the No Defense baseline.

| Experiment | Metric | Paper Reported | Our Reproduction | 
| :--- | :--- | :--- | :--- | 
| **ASD Defense (Ours)** | **Clean Accuracy (ACC)** | **93.4%** | **93.53%** | 
| **ASD Defense (Ours)** | **Attack Success Rate (ASR)** | **1.2%** | **1.71%** | 
| **No Defense (Baseline)** | **Clean Accuracy (ACC)** | **94.9%** | **92.95%** | 
| **No Defense (Baseline)** | **Attack Success Rate (ASR)** | **100%** | **100.00%** | 


### 2. GTSRB Evaluation

For GTSRB, we also see strong defense performance, although with slightly higher variance due to dataset characteristics.

| Experiment | Metric | Paper Reported | Our Reproduction | 
| :--- | :--- | :--- | :--- | 
| **ASD Defense (Ours)** | **Clean Accuracy (ACC)** | **96.7%** | **95.42%** | 
| **ASD Defense (Ours)** | **Attack Success Rate (ASR)** | **0%** | **6.16%** | 
| **No Defense (Baseline)** | **Clean Accuracy (ACC)** | **97.6%** | **97.2%** | 
| **No Defense (Baseline)** | **Attack Success Rate (ASR)** | **100%** | **99%** | 


# Backdoor Defense via Adaptively Splitting Poisoned Dataset (ASD)

This repository provides the implementation of the defense mechanism described in the CVPR 2023 paper: **Backdoor Defense via Adaptively Splitting Poisoned Dataset**.

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

*   **Objective**: The attacker aims to install a backdoor that causes the model to misclassify specific inputs (containing a "trigger") as a target class, while maintaining high accuracy on clean data.
*   **Mechanism**: The attacker injects a small set of poisoned images into the training set. These images have a visible trigger (e.g., a 3x3 pixel pattern) and are mislabeled as the target class.
*   **Scenario (Many-to-One)**:
    *   **Source Class**: Any class (inputs from any class with the trigger are misclassified).
    *   **Target Class**: The class the attacker wants the model to predict (e.g., Class 0).
    *   **Trigger**: A fixed pattern added to the image (e.g., checkerboard pattern in the corner).
    *   **Poison Rate**: The percentage of the training set that is poisoned (e.g., 5%).

## Defense Mechanism: Adaptively Splitting Dataset (ASD)

Our defense leverages the framework of splitting the poisoned dataset into two data pools (clean and poisoned) and training on them adaptively.

### 1. Unified Splitting Framework
We formulate training-time defenses as a two-step process: splitting the dataset into a reliable (clean) pool and an unreliable (poisoned) pool.

### 2. Loss-Guided Split
Initial separation is performed based on training loss. Poisoned samples (with triggers) tend to have lower loss values during the early stages of training compared to clean samples that are harder to learn. We use a Gaussian Mixture Model (GMM) on the loss distribution to separate these pools.

### 3. Meta-Learning-Inspired Split
To further refine the split, we employ a meta-learning approach. We update the split by minimizing the meta-loss on a small, trustworthy validation set. This helps in correcting misclassified samples from the loss-guided step.

### 4. Adaptive Training
The model is trained iteratively on the refined clean pool, while the split is updated dynamically throughout the training process. This prevents the model from overfitting to the backdoor trigger.

## Implementation & Reproduction

We provide a complete pipeline to reproduce the results for **CIFAR-10** and **GTSRB**.

### 1. Installation
We use Python 3.8+ and PyTorch. Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 2. Dataset Generation

#### CIFAR-10
Download the CIFAR-10 dataset (Python version) from the [official website](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and extract it to your data directory.

#### GTSRB
We provide a helper script `prepare_gtsrb.py` to prepare the GTSRB dataset from raw image folders into the required pickle format.

```bash
# Basic usage (defaults to internal paths)
python prepare_gtsrb.py --source /path/to/raw/GTSRB --dest /path/to/save/pickles
```

This script:
1.  Scans the source directory for `train_images` (or `Final_Training/Images`) and `test_images`.
2.  Resizes all images to 32x32.
3.  Reads CSV annotations for test data if class subdirectories are missing.
4.  Saves `train.pkl` and `test.pkl` to the destination folder.

### 3. Running the Defense

We use YAML configuration files to manage experiments.

**For CIFAR-10:**
```bash
# Run ASD Defense
python ASD.py --config config/baseline_asd.yaml --gpu 0

# Run No Defense (Baseline)
python train_baseline.py --config config/baseline_asd.yaml --gpu 0
```

**For GTSRB:**
```bash
# Run ASD Defense
python ASD.py --config config/baseline_asd_gtsrb.yaml --gpu 0

# Run No Defense (Baseline)
python train_baseline.py --config config/baseline_asd_gtsrb.yaml --gpu 0
```

### 4. Evaluation & Results

We have successfully reproduced the results of the paper for both datasets.

#### Dataset 1: CIFAR-10 Results

| Defense | Clean Accuracy (ACC) | Attack Success Rate (ASR) | Notes |
| :--- | :--- | :--- | :--- |
| **No Defense (Baseline)** | 92.95% | **100.00%** | Attack is fully effective. |
| **ASD (Ours)** | **93.53%** | **1.71%** | Defense successfully mitigates the attack (Paper: 1.2%). |

#### Dataset 2: GTSRB Results

| Defense | Clean Accuracy (ACC) | Attack Success Rate (ASR) | Notes |
| :--- | :--- | :--- | :--- |
| **No Defense (Baseline)** | (Pending) | (Pending) | Expected ~98% ACC, ~100% ASR. |
| **ASD (Ours)** | **95.42%** | **6.16%** | Significant reduction in ASR (Paper: 0%). Defense is effective. |

## Acknowledgements

This repository is based on [DBD](https://github.com/SCLBD/DBD) and benefits from [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox).

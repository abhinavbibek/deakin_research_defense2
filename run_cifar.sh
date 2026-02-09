#!/bin/bash

## Train ASD model under BadNets attack for CIFAR10
python ASD.py \
  --config config/baseline_asd_cifar10.yaml \
  --resume False \
  --gpu 1


## Test ASD model under BadNets attack for CIFAR10
python test.py --config config/baseline_asd_cifar10.yaml --resume latest_model.pt --gpu 1


## Training No Defense model under BadNets attack for CIFAR10
python train_baseline.py --config config/baseline_asd_cifar10.yaml --gpu 1

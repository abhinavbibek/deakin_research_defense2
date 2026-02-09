#!/bin/bash

## Train ASD model under BadNets attack for GTSRB
python ASD.py --config config/baseline_asd_gtsrb.yaml --resume False --gpu 1

## Test ASD model under BadNets attack for GTSRB
python test.py --config config/baseline_asd_gtsrb.yaml --resume latest_model.pt --gpu 1


## Training No Defense model under BadNets attack for GTSRB
python train_baseline.py --config config/baseline_asd_gtsrb.yaml --gpu 0

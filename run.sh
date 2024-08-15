#!/bin/bash

set -x

python fine_tuning.py fit --config configs/dino_vitb16_n_splits_3-no_mix-lr_005.yaml
python fine_tuning.py fit --config configs/dino_vitb16-n_splits_1-mix-lr_005.yaml
python fine_tuning.py fit --config configs/dino_vitb16-n_splits_3-mix-lr_005.yaml

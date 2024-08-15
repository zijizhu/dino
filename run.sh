#!/bin/bash

CONFIG_DIR="configs/*"

for fn in $CONFIG_DIR
do
    python fine_tuning.py fit --config "$fn"
done

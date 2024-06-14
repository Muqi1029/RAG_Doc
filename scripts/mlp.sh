#!/usr/bin/env bash

python run.py --loss 'nce' --epoch 300 --lr 0.08

python run.py --loss 'nce' --test

echo "=============="
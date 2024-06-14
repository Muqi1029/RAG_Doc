#!/usr/bin/env bash

python run.py --loss 'nce' --epoch 1 --lr 1

python run.py --loss 'nce' --test

#!/usr/bin/env bash

python run.py --model simclr --epochs 200 --lr 0.01
python run.py --model simclr --test

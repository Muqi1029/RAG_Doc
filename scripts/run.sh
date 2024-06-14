#!/usr/bin/env bash

epoch=300

python run.py --loss 'nce' --epoch ${epoch} --lr 0.08 --random

python run.py --loss 'nce' --test --random

python run.py --loss 'nce' --epoch ${epoch} --lr 0.08

python run.py --loss 'nce' --test


echo "=============="

python run.py --model simclr --epochs ${epoch} --lr 0.001

python run.py --model simclr --test

python run.py --model simclr --epochs ${epoch} --lr 0.001 --random

python run.py --model simclr --test --random
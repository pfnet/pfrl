#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gym/reinforce
python examples/gym/train_reinforce_gym.py --steps 100 --batchsize 1 --outdir $outdir/gym/reinforce --gpu $gpu
model=$(find $outdir/gym/reinforce -name "*_finish")
python examples/gym/train_reinforce_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

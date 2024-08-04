#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gymnasium/reinforce
python examples/gymnasium/train_reinforce_gymnasium.py --steps 100 --batchsize 1 --outdir $outdir/gymnasium/reinforce --gpu $gpu
model=$(find $outdir/gymnasium/reinforce -name "*_finish")
python examples/gymnasium/train_reinforce_gymnasium.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

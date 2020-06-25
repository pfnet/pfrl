#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atari/ppo
python examples/atari/train_ppo_ale.py --env PongNoFrameskip-v4 --steps 100 --update-interval 50 --batchsize 16 --epochs 2 --outdir $outdir/atari/ppo --gpu $gpu
model=$(find $outdir/atari/ppo -name "*_finish")
python examples/atari/train_ppo_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

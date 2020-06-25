#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gym/dqn
python examples/gym/train_dqn_gym.py --steps 100 --replay-start-size 50 --outdir $outdir/gym/dqn --gpu $gpu
model=$(find $outdir/gym/dqn -name "*_finish")
python examples/gym/train_dqn_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

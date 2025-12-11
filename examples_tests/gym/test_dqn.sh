#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gymnasium/dqn
python examples/gymnasium/train_dqn_gymnasium.py --steps 100 --replay-start-size 50 --outdir $outdir/gymnasium/dqn --gpu $gpu
model=$(find $outdir/gymnasium/dqn -name "*_finish")
python examples/gymnasium/train_dqn_gymnasium.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

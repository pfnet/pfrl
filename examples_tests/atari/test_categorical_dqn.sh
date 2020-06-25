#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atari/categorical_dqn
python examples/atari/train_categorical_dqn_ale.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/atari/categorical_dqn --gpu $gpu
model=$(find $outdir/atari/categorical_dqn -name "*_finish")
python examples/atari/train_categorical_dqn_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

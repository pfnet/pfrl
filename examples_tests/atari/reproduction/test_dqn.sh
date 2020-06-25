#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atari/reproduction/dqn
python examples/atari/reproduction/dqn/train_dqn.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/atari/reproduction/dqn --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1 --gpu $gpu
model=$(find $outdir/atari/reproduction/dqn -name "*_finish")
python examples/atari/reproduction/dqn/train_dqn.py --env PongNoFrameskip-v4 --demo --load $model --outdir $outdir/temp --eval-n-steps 200 --gpu $gpu

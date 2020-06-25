#!/bin/bash

set -Ceu

outdir=$(mktemp -d)
echo "outdir: $outdir"

gpu="$1"

# atari/reproduction/rainbow
python examples/atari/reproduction/rainbow/train_rainbow.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/atari/reproduction/rainbow --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1 --gpu $gpu
model=$(find $outdir/atari/reproduction/rainbow -name "*_finish")
python examples/atari/reproduction/rainbow/train_rainbow.py --env PongNoFrameskip-v4 --demo --load $model --outdir $outdir/temp --eval-n-steps 200 --gpu $gpu

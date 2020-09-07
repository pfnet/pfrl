#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atari/double_iqn
python examples/atari/train_double_iqn.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/atari/double_iqn --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1  --gpu $gpu
model=$(find $outdir/atari/double_iqn -name "*_finish")
python examples/atari/train_double_iqn.py --env PongNoFrameskip-v4 --demo --load $model --outdir $outdir/temp --eval-n-steps 200 --gpu $gpu

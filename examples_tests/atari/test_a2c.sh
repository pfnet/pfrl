#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atari/a2c
python examples/atari/train_a2c_ale.py --env PongNoFrameskip-v4 --steps 100 --update-steps 50 --outdir $outdir/atari/a2c
model=$(find $outdir/atari/a2c -name "*_finish")
python examples/atari/train_a2c_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1

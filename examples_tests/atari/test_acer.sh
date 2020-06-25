#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atari/acer (only for cpu)
if [[ $gpu -lt 0 ]]; then
  python examples/atari/train_acer_ale.py 4 --env PongNoFrameskip-v4 --steps 100 --outdir $outdir/atari/acer
  model=$(find $outdir/atari/acer -name "*_finish")
  python examples/atari/train_acer_ale.py 4 --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp
fi

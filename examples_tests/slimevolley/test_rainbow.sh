#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# slimevolley/rainbow
# Use CartPole-v0 to test without installing slimevolleygym
python examples/slimevolley/train_rainbow.py --gpu $gpu --steps 100 --outdir $outdir/slimevolley/rainbow --env CartPole-v0
model=$(find $outdir/slimevolley/rainbow -name "*_finish")
python examples/slimevolley/train_rainbow.py --demo --load $model --eval-n-episodes 1 --outdir $outdir/temp --gpu $gpu --env CartPole-v0

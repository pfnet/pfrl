#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# her/dqn_bit_flip
python examples/her/train_dqn_bit_flip.py --gpu $gpu --steps 100 --outdir $outdir/her/bit_flip
model=$(find $outdir/her/bit_flip -name "*_finish")
python examples/her/train_dqn_bit_flip.py --demo --load $model --eval-n-episodes 1 --outdir $outdir/temp --gpu $gpu
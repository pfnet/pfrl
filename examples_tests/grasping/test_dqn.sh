#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# grasping/dqn
python examples/grasping/train_dqn_batch_grasping.py --gpu $gpu --steps 100 --outdir $outdir/grasping/dqn
model=$(find $outdir/grasping/dqn -name "*_finish")
python examples/grasping/train_dqn_batch_grasping.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

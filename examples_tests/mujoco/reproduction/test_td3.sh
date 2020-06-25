#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# mujoco/reproduction/td3 (specify non-mujoco env to test without mujoco)
python examples/mujoco/reproduction/td3/train_td3.py --env Pendulum-v0 --gpu $gpu --steps 10 --replay-start-size 5 --batch-size 5 --outdir $outdir/mujoco/reproduction/td3
model=$(find $outdir/mujoco/reproduction/td3 -name "*_finish")
python examples/mujoco/reproduction/td3/train_td3.py --env Pendulum-v0 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

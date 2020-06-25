#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# mujoco/reproduction/trpo (specify non-mujoco env to test without mujoco)
python examples/mujoco/reproduction/trpo/train_trpo.py --steps 10 --trpo-update-interval 5 --outdir $outdir/mujoco/reproduction/trpo --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/mujoco/reproduction/trpo -name "*_finish")
python examples/mujoco/reproduction/trpo/train_trpo.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/temp --gpu $gpu

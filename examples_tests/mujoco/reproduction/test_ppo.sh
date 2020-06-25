#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# mujoco/reproduction/ppo (specify non-mujoco env to test without mujoco)
python examples/mujoco/reproduction/ppo/train_ppo.py --steps 10 --update-interval 5 --batch-size 5 --epochs 2 --outdir $outdir/mujoco/reproduction/ppo --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/mujoco/reproduction/ppo -name "*_finish")
python examples/mujoco/reproduction/ppo/train_ppo.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/temp --gpu $gpu

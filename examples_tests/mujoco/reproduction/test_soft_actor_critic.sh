#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# mujoco/soft_actor_critic (specify non-mujoco env to test without mujoco)
python examples/mujoco/reproduction/soft_actor_critic/train_soft_actor_critic.py --env Pendulum-v0 --gpu $gpu --steps 10 --replay-start-size 5 --batch-size 5 --outdir $outdir/mujoco/soft_actor_critic
model=$(find $outdir/mujoco/soft_actor_critic -name "*_finish")
python examples/mujoco/reproduction/soft_actor_critic/train_soft_actor_critic.py --env Pendulum-v0 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

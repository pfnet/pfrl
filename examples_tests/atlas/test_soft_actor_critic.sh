#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atlas/soft_actor_critic
# Use Pendulum-v0 to test without installing roboschool
python examples/atlas/train_soft_actor_critic_atlas.py --gpu $gpu --steps 100 --outdir $outdir/atlas/soft_actor_critic --env Pendulum-v0
model=$(find $outdir/atlas/soft_actor_critic -name "*_finish")
python examples/atlas/train_soft_actor_critic_atlas.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu --env Pendulum-v0

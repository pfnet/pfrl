#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# optuna/dqn_lunarlander
storage="sqlite:///${outdir}/tmp.db"
study="optuna-pfrl-test"
optuna create-study --study-name "${study}" --storage "${storage}" --direction maximize

python examples/optuna/optuna_dqn_lunarlander.py \
  -optuna-study-name "${study}" --optuna-storage "${storage}" --optuna-n-trials 3 \
  --steps 100 --gpu $gpu --outdir $outdir/optuna/dqn_lunarlander

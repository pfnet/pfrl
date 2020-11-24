#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"
storage="sqlite:///${outdir}/tmp.db"

# optuna/dqn_obs1d/NopPruner
study="optuna-pfrl-test-nop"
optuna create-study --study-name "${study}" --storage "${storage}" --direction maximize

python examples/optuna/optuna_dqn_obs1d.py \
  --env "CartPole-v0" \
  --optuna-study-name "${study}" --optuna-storage "${storage}" --optuna-training-steps-budget 300 \
  --optuna-pruner "NopPruner" \
  --eval-interval 30 --eval-n-episodes 2 \
  --steps 100 --gpu $gpu --outdir $outdir/optuna/dqn_obs1d/NopPruner

# optuna/dqn_obs1d/ThresholdPruner
study="optuna-pfrl-test-threshold"
optuna create-study --study-name "${study}" --storage "${storage}" --direction maximize

python examples/optuna/optuna_dqn_obs1d.py \
  --env "CartPole-v0" \
  --optuna-study-name "${study}" --optuna-storage "${storage}" --optuna-training-steps-budget 300 \
  --optuna-pruner "ThresholdPruner" --lower=-10 \
  --eval-interval 30 --eval-n-episodes 2 \
  --steps 100 --gpu $gpu --outdir $outdir/optuna/dqn_obs1d/ThresholdPruner

# optuna/dqn_obs1d/PercentilePruner
study="optuna-pfrl-test-percentile"
optuna create-study --study-name "${study}" --storage "${storage}" --direction maximize

python examples/optuna/optuna_dqn_obs1d.py \
  --env "CartPole-v0" \
  --optuna-study-name "${study}" --optuna-storage "${storage}" --optuna-training-steps-budget 300 \
  --optuna-pruner "PercentilePruner" --percentile 25.0 --n-startup-trials 1 --n-warmup-steps 50 \
  --eval-interval 30 --eval-n-episodes 2 \
  --steps 100 --gpu $gpu --outdir $outdir/optuna/dqn_obs1d/PercentilePruner

# optuna/dqn_obs1d/HyperbandPruner
study="optuna-pfrl-test-hyperband"
optuna create-study --study-name "${study}" --storage "${storage}" --direction maximize

python examples/optuna/optuna_dqn_obs1d.py \
  --env "CartPole-v0" \
  --optuna-study-name "${study}" --optuna-storage "${storage}" --optuna-training-steps-budget 300 \
  --optuna-pruner "HyperbandPruner" \
  --eval-interval 30 --eval-n-episodes 2 \
  --steps 100 --gpu $gpu --outdir $outdir/optuna/dqn_obs1d/HyperbandPruner

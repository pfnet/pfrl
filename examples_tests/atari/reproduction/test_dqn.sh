#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atari/reproduction/dqn
python examples/atari/reproduction/dqn/train_dqn.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/atari/reproduction/dqn --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1 --gpu $gpu
model=$(find $outdir/atari/reproduction/dqn -name "*_finish")
python examples/atari/reproduction/dqn/train_dqn.py --env PongNoFrameskip-v4 --demo --load $model --outdir $outdir/temp --eval-n-steps 200 --gpu $gpu

# snapshot without eval
python examples/atari/reproduction/dqn/train_dqn.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/atari/reproduction/dqn --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1 --gpu $gpu --exp-id 0 --save-snapshot --checkpoint-freq 45
python examples/atari/reproduction/dqn/train_dqn.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/atari/reproduction/dqn --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1 --gpu $gpu --exp-id 0 --load-snapshot

# snapshot after eval
python examples/atari/reproduction/dqn/train_dqn.py --env PongNoFrameskip-v4 --steps 4600 --replay-start-size 50 --outdir $outdir/atari/reproduction/dqn --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1 --gpu $gpu --exp-id 1 --save-snapshot --checkpoint-freq 4000
python examples/atari/reproduction/dqn/train_dqn.py --env PongNoFrameskip-v4 --steps 4700 --replay-start-size 50 --outdir $outdir/atari/reproduction/dqn --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1 --gpu $gpu --exp-id 1 --load-snapshot

# TRPO on MuJoCo benchmarks

This example trains a TRPO agent ([Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)) on MuJoCo benchmarks from OpenAI Gym.

We follow the training and evaluation settings of [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), which provides thorough, highly tuned benchmark results.

## Requirements

- MuJoCo Pro 1.5
- mujoco_py>=1.50, <2.1

## Running the Example

To run the training example:
```
python train_trpo.py [options]
```

We have already pretrained models from this script for all the domains listed in the [results](#Results) section. To load a pretrained model:

```
python train_trpo.py --demo --load-pretrained --env HalfCheetah-v2 --pretrained-type best --gpu -1
```

### Useful Options

- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_trpo.py --gpu -1`.
- `--env`. Specifies the environment. E.g. `python train_trpo.py --env HalfCheetah-v2`.
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.
- `--demo`. Runs an evaluation, instead of training the agent.
- `--load-pretrained` Loads the pretrained model. Both `--load` and `--load-pretrained` cannot be used together.
- `--pretrained-type`. Either `best` (the best intermediate network during training) or `final` (the final network after training).

To view the full list of options, either view the code or run the example with the `--help` option.

## Known differences

- We used version v2 of the environments whereas the original results were reported for version v1, however this doesn't seem to introduce significant differences: https://github.com/openai/gym/pull/834

## Results

These scores are evaluated by average return +/- standard error of 100 evaluation episodes after 2M training steps.

Reported scores are taken from the row Table 1 of [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560).
Here we try to reproduce TRPO (Schulman et al. 2017) of the (64, 64) column, which corresponds to the default settings.

PFRL scores are based on 20 trials using different random seeds, using the following command.

```
python train_trpo.py --seed [0-19] --env [env]
```

| Environment | PFRL Score | Reported Score |
| ----------- |:---------------:|:--------------:|
| HalfCheetah |  **1561**+/-114 |      205+/-256 |
| Hopper      |   **3038**+/-42 |      2828+/-70 |
| Walker2d    |       3260+/-82 |            N/A |
| Swimmer     |        197+/-26 |            N/A |

### Learning Curves

The shaded region represents a standard deviation of the average evaluation over 20 trials.

![HalfCheetah-v2](assets/HalfCheetah-v2.png)
![Hopper-v2](assets/Hopper-v2.png)
![Walker2d-v2](assets/Walker2d-v2.png)
![Swimmer-v2](assets/Swimmer-v2.png)

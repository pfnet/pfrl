# PPO on MuJoCo benchmarks

This example trains a PPO agent ([Proximal Policy Optimization Algorithms](http://arxiv.org/abs/1707.06347)) on MuJoCo benchmarks from OpenAI Gym.

We follow the training and evaluation settings of [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), which provides thorough, highly tuned benchmark results.

## Requirements

- MuJoCo Pro 1.5
- mujoco_py>=1.50, <2.1

## Running the Example

To run the training example:
```
python train_ppo.py [options]
```

We have already pretrained models from this script for all the domains listed in the [results](#Results) section. To load a pretrained model:

```
python train_ppo.py --demo --load-pretrained --env HalfCheetah-v2 --gpu -1
```

### Useful Options

- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_ppo.py --gpu -1`.
- `--env`. Specifies the environment. E.g. `python train_ppo.py --env HalfCheetah-v2`.
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.
- `--demo`. Runs an evaluation, instead of training the agent.
- `--load-pretrained` Loads the pretrained model. Both `--load` and `--load-pretrained` cannot be used together.

To view the full list of options, either view the code or run the example with the `--help` option.

## Known differences

- While the original paper initialized weights by normal distribution (https://github.com/Breakend/baselines/blob/50ffe01d254221db75cdb5c2ba0ab51a6da06b0a/baselines/ppo1/mlp_policy.py#L28), we use orthogonal initialization as the latest openai/baselines does (https://github.com/openai/baselines/blob/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/a2c/utils.py#L61).
- We used version v2 of the environments whereas the original results were reported for version v1, however this doesn't seem to introduce significant differences: https://github.com/openai/gym/pull/834

## Results

These scores are evaluated by average return +/- standard error of 100 evaluation episodes after 2M training steps.

Reported scores are taken from Table 1 of [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560).

PFRL scores are based on 20 trials using different random seeds, using the following command.

```
python train_ppo.py --gpu 0 --seed [0-19] --env [env]
```

| Environment | PFRL Score      | Reported Score |
| ----------- |:---------------:|:--------------:|
| HalfCheetah |      1986+/-175 | **2201**+/-323 |
| Hopper      |   **2793**+/-65 |      2790+/-62 |
| Walker2d    |      2873+/-138 |            N/A |
| Swimmer     |        125+/-11 |            N/A |


### Training times
These training times were obtained by running `train_ppo.py` on a single CPU and a single GPU.


| Environment | PFRL Time (hours) |
| ----------- |:-----------------:|
| HalfCheetah | 2.75              |
| Hopper      | 2.75              |
| Swimmer     | 2.77              |
| Walker2d    | 2.89              |


### Learning Curves

The shaded region represents a standard deviation of the average evaluation over 20 trials.

![HalfCheetah-v2](assets/HalfCheetah-v2.png)
![Hopper-v2](assets/Hopper-v2.png)
![Walker2d-v2](assets/Walker2d-v2.png)
![Swimmer-v2](assets/Swimmer-v2.png)

# TD3 on MuJoCo benchmarks

This example trains a TD3 agent ([Addressing Function Approximation Error in Actor-Critic Methods](http://arxiv.org/abs/1802.09477)) on MuJoCo benchmarks from OpenAI Gym.

## Requirements

- MuJoCo Pro 1.5
- mujoco_py>=1.50, <2.1

## Running the Example

To run the training example:
```
python train_td3.py [options]
```

We have already pretrained models from this script for all the domains listed in the [results](#Results) section. To load a pretrained model:

```
python train_td3.py --demo --load-pretrained --env HalfCheetah-v2 --pretrained-type best --gpu -1
```


### Useful Options

- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_td3.py --gpu -1`.
- `--env`. Specifies the environment. E.g. `python train_td3.py --env HalfCheetah-v2`.
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.
- `--demo`. Runs an evaluation, instead of training the agent.
- `--load-pretrained` Loads the pretrained model. Both `--load` and `--load-pretrained` cannot be used together.
- `--pretrained-type`. Either `best` (the best intermediate network during training) or `final` (the final network after training).

To view the full list of options, either view the code or run the example with the `--help` option.

## Known differences

- While the original paper used different numbers of timesteps for which purely random policy is used for different environments (10000 for HalfCheetah and Ant, 1000 for the remaining), we used 10000 across all the environments.
- While [the original implementation](https://github.com/sfujim/TD3) updates the model N times after every episode where N is the length of the episode, our implementation updates the model once every step. In addition, ours does not update the model until the replay buffer has at least 10000 transitions.

## Results

PFRL scores are based on 10 trials using different random seeds, using the following command.

```
python train_td3.py --seed [0-9] --env [env]
```

During each trial, the agent is trained for 1M timesteps and evaluated after every 5000 timesteps, resulting in 200 evaluations.
Each evaluation reports average return over 10 episodes without exploration noise.

### Max Average Return

Maximum evaluation scores, averaged over 10 trials (+/- standard deviation), are reported for each environment.

Reported scores are taken from the "TD3" column of Table 1 of [Addressing Function Approximation Error in Actor-Critic Methods](http://arxiv.org/abs/1802.09477).
Although the original paper used v1 versions of MuJoCo envs, we used v2 as v1 are not supported by recent versions of OpenAI Gym.

| Environment               | PFRL Score            | Reported Score        |
| ------------------------- |:---------------------:|:---------------------:|
| HalfCheetah-v2            | **10417.28**+/-833.17 |     9636.95+/-859.065 |
| Hopper-v2                 |  **3691.14**+/-138.22 |      3564.07+/-114.74 |
| Walker2d-v2               |      4430.85+/-667.57 |  **4682.82**+/-539.64 |
| Ant-v2                    |  **4752.56**+/-992.02 |     4372.44+/-1000.33 |
| Reacher-v2                |      **-2.69**+/-0.08 |          -3.60+/-0.56 |
| InvertedPendulum-v2       |        1000.00+/-0.00 |        1000.00+/-0.00 |
| InvertedDoublePendulum-v2 |     7507.61+/-3903.04 |   **9337.47**+/-14.96 |


### Last 100 Average Return

Average return of last 10 evaluation scores, averaged over 10 trials, are reported for each environment.

Reported scores are taken from the "TD3" row of Table 2 of [Addressing Function Approximation Error in Actor-Critic Methods](http://arxiv.org/abs/1802.09477).
Although the original paper used v1 versions of MuJoCo envs, we used v2 as v1 are not supported by recent versions of OpenAI Gym.

| Environment               | PFRL Score   | Reported Score |
| ------------------------- |:------------:|:--------------:|
| HalfCheetah-v2            | **10047.79** |        9532.99 |
| Hopper-v2                 |  **3400.06** |        3304.75 |
| Walker2d-v2               |      4056.17 |    **4565.24** |
| Ant-v2                    |  **4333.70** |        4185.06 |
| Reacher-v2                |        -3.87 |            N/A |
| InvertedPendulum-v2       |       964.92 |            N/A |
| InvertedDoublePendulum-v2 |      7216.69 |            N/A |

### Training times
These training times were obtained by running `train_td3.py` on a single CPU and a single GPU.

| Environment            | PFRL Time (hours) |
| ---------------------- |:-----------------:|
| HalfCheetah            | 3.29              |
| Hopper                 | 3.27              |
| Walker2d               | 3.34              |
| Ant                    | 3.64              |
| Reacher                | 3.03              |
| InvertedPendulum       | 3.21              |
| InvertedDoublePendulum | 3.21              |


### Learning Curves

The shaded region represents a standard deviation of the average evaluation over 10 trials.

![HalfCheetah-v2](assets/HalfCheetah-v2.png)
![Hopper-v2](assets/Hopper-v2.png)
![Walker2d-v2](assets/Walker2d-v2.png)
![Ant-v2](assets/Ant-v2.png)
![Reacher-v2](assets/Reacher-v2.png)
![InvertedPendulum-v2](assets/InvertedPendulum-v2.png)
![InvertedDoublePendulum-v2](assets/InvertedDoublePendulum-v2.png)

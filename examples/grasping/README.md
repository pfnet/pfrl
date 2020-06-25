# Bullet-based robotic grasping

This directory contains example scripts that learn to grasp objects in an environment simulated by Bullet, a physics simulator.

![Grasping](../../assets/grasping.gif)

## Files

- `train_dqn_batch_grasping.py`: DoubleDQN + prioritized experience replay

## Requirements

- pybullet>=2.4.9

## How to run

Train with one simulator, which is slow.
```
python examples/grasping/train_dqn_batch_grasping.py
```

Train with 96 simulators run in parallel, which is faster.
```
python examples/grasping/train_dqn_batch_grasping.py --num-envs 96
```

Watch how the learned agent performs. `<path to agent>` must be a path to a directory where the agent was saved (e.g. `2000000_finish` created inside the output directory specified as `--outdir`).
```
python examples/grasping/train_dqn_batch_grasping.py --demo --render --load <path to agent>
```

### Useful Options
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_dqn_batch_grasping.py --gpu -1`.
- `--num-envs` Specifies the number of parallel environments to spawn, e.g. `--num-envs 96`
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.

## Results

Below is the learning curve of the example script with 24 CPUs and a single GPU, using 96 environments, averaged over three trials with different random seeds. Each trial took around 38 hours for 20M steps.

The highest achieved grasping performance during training is 84% (with `--seed 0`), where a success corresponds to a successful grasp of any object.

![LearningCurve](assets/learningcurve.png)


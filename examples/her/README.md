# Hindsight Experience Replay
These two examples train agents using [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495). The first example, `train_dqn_bit_flip.py` trains a DoubleDQN in the BitFlip environment as described in the HER paper. The second example, `train_ddpg_her_fetch.py` trains agents in the robotic Fetch environments, also described in the HER paper.

## To Run:

To run the bitflip example:
```
python train_dqn_bit_flip.py --num-bits <number of bits>
```

To run DDPG with HER on fetch tasks, run:
```
python train_ddpg_her_fetch.py --env <Gym environment name>
```

Options
- `--gpu`: Set to -1 if you have no GPU.

## Results and Reproducibility
The BitFlip environment was implemented as per the description in the paper. The DQN algorithm for the bitflip environment is not from the paper (to our knowledge there is no publicly released implementation).

For the Fetch environments, we added an action penalty, return clipping, and observation normalization to DDPG as done by the [OpenAI baselines implementation](https://github.com/openai/baselines/tree/master/baselines/her).


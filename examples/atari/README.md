# Examples for Arcade Learning Environment

- `train_a2c_ale.py`: A2C
- `train_acer_ale.py`: ACER
- `train_categorical_dqn_ale.py`: CategoricalDQN
- `train_dqn_ale.py`: DQN, DoubleDQN or PAL
- `train_dqn_batch_ale.py`: DQN, DoubleDQN or PAL on batched env
- `train_drqn_ale.py`: DRQN (DQN with a recurrent model)
- `train_ppo_ale.py`: PPO

## Requirements

- atari_py>=0.1.1
- opencv-python

## How to run

```
python train_a2c_ale.py [options]
python train_acer_ale.py n_processes [options]
python train_categorical_dqn_ale.py [options]
python train_dqn_ale.py [options]
python train_dqn_batch_ale.py [options]
python train_drqn_ale.py [options]
python train_ppo_ale.py [options]
```

Specify `--help` or read code for options.

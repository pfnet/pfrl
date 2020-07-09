# IQN
This example trains an IQN agent, from the following paper: [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923). 

## Requirements

- atari_py>=0.1.1
- opencv-python

## Running the Example

To run the training example:
```
python train_iqn.py [options]
```

We have already pretrained models from this script for all the domains listed in the [results](#Results). Note that while we may have run multiple seeds, our pretrained model represents a single run from this script, and may not be achieve the performance of the [results](#Results). To load a pretrained model:

```
python train_iqn.py --demo --load-pretrained --env BreakoutNoFrameskip-v4 --pretrained-type best --gpu -1
```

### Useful Options
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_dqn.py --gpu -1`.
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.
- `--demo`. Runs an evaluation, instead of training the agent.
- `--load-pretrained` Loads the pretrained model. Both `--load` and `--load-pretrained` cannot be used together.
- `--pretrained-type`. Either `best` (the best intermediate network during training) or `final` (the final network after training).

To view the full list of options, either view the code or run the example with the `--help` option.


## Results
These results reflect PFRL commit hash: `a0ad6a7`. We use the same evaluation protocol used in the IQN paper.


| Results Summary ||
| ------------- |:-------------:|
| Reporting Protocol | The highest mean intermediate evaluation score |
| Number of seeds | 3 |
| Number of common domains | 55 |
| Number of domains where paper scores higher | 23 |
| Number of domains where PFRL scores higher | 26 |
| Number of ties between paper and PFRL | 6 | 


| Game        | PFRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 10124.5| N/A|
| Alien | **11625.6**| 7022|
| Amidar | 1984.1| **2946**|
| Assault | 23126.2| **29091**|
| Asterix | **485221.5**| 342016|
| Asteroids | **3662.0**| 2898|
| Atlantis | 939633.3| **978200**|
| BankHeist | 1338.2| **1416**|
| BattleZone | **61428.6**| 42244|
| BeamRider | 35294.8| **42776**|
| Berzerk | **2295.6**| 1053|
| Bowling | **94.5**| 86.5|
| Boxing | **99.9**| 99.8|
| Breakout | 708.0| **734**|
| Carnival | 5836.8| N/A|
| Centipede | 10702.5| **11561**|
| ChopperCommand | **23182.5**| 16836|
| CrazyClimber | 172486.1| **179082**|
| DemonAttack | **132576.7**| 128580|
| DoubleDunk | -0.1| **5.6**|
| Enduro | 2359.0| 2359|
| FishingDerby | **43.8**| 33.8|
| Freeway | 34.0| 34.0|
| Frostbite | **7820.7**| 4342|
| Gopher | 112949.3| **118365**|
| Gravitar | **1045.9**| 911|
| Hero | 25299.7| **28386**|
| IceHockey | **2.5**| 0.2|
| Jamesbond | 26284.1| **35108**|
| JourneyEscape | -640.8| N/A|
| Kangaroo | 15014.8| **15487**|
| Krull | 9625.8| **10707**|
| KungFuMaster | **85625.3**| 73512|
| MontezumaRevenge | 0.0| 0.0|
| MsPacman | 4818.8| **6349**|
| NameThisGame | 22553.7| **22682**|
| Phoenix | **138020.8**| 56599|
| Pitfall | 0.0| 0.0|
| Pong | 21.0| 21.0|
| Pooyan | 18799.4| N/A|
| PrivateEye | **1685.7**| 200|
| Qbert | **26133.3**| 25750|
| Riverraid | **21663.9**| 17765|
| RoadRunner | **66602.9**| 57900|
| Robotank | **76.2**| 62.5|
| Seaquest | 27528.7| **30140**|
| Skiing | **-9256.7**| -9289|
| Solaris | 7606.3| **8007**|
| SpaceInvaders | **30784.6**| 28888|
| StarGunner | **172230.3**| 74677|
| Tennis | 23.6| 23.6|
| TimePilot | 11648.8| **12236**|
| Tutankham | **345.1**| 293|
| UpNDown | 84747.0| **88148**|
| Venture | 1027.1| **1318**|
| VideoPinball | **714777.3**| 698045|
| WizardOfWor | 24954.3| **31190**|
| YarsRevenge | **29202.1**| 28379|
| Zaxxon | 17905.8| **21772**|


## Evaluation Protocol
Our evaluation protocol is designed to mirror the evaluation protocol of the original paper as closely as possible, in order to offer a fair comparison of the quality of our example implementation. Specifically, the details of our evaluation (also can be found in the code) are the following:

- **Evaluation Frequency**: The agent is evaluated after every 1 million frames (250K timesteps). This results in a total of 200 "intermediate" evaluations.
- **Evaluation Phase**: The agent is evaluated for 500K frames (125K timesteps) in each intermediate evaluation. 
	- **Output**: The output of an intermediate evaluation phase is a score representing the mean score of all completed evaluation episodes within the 125K timesteps. If there is any unfinished episode by the time the 125K timestep evaluation phase is finished, that episode is discarded.
- **Intermediate Evaluation Episode**: 
	- Capped at 30 mins of play, or 108K frames/ 27K timesteps.
	- Each evaluation episode begins with a random number of no-ops (up to 30), where this number is chosen uniformly at random.
- **Reporting**: For each run of our IQN example, we take the best outputted score of the intermediate evaluations to be the evaluation for that agent. We then average this over all runs (i.e. seeds) to produce the output reported in the table.


## Training times

| Training time (in days) across all runs (# domains x # seeds) | |
| ------------- |:-------------:|
| Mean        |  4.866 |
| Standard deviation | 0.152|
| Fastest run | 4.472|
| Slowest run | 5.295|




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

### Useful Options
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_dqn.py --gpu -1`.
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.
- `--demo`. Runs an evaluation, instead of training the agent.
- (Currently unsupported) `--load-pretrained` Loads the pretrained model. Both `--load` and `--load-pretrained` cannot be used together.
- `--pretrained-type`. Either `best` (the best intermediate network during training) or `final` (the final network after training).

To view the full list of options, either view the code or run the example with the `--help` option.


## Results
**NOTE: These results reflect scores from our predecessor library, ChainerRL. We will release benchmarks for PFRL in the future.**

These results reflect ChainerRL  `v0.6.0`. We use the same evaluation protocol used in the IQN paper.


| Results Summary ||
| ------------- |:-------------:|
| Number of seeds | 2 |
| Number of common domains | 55 |
| Number of domains where paper scores higher | 25 |
| Number of domains where ChainerRL scores higher | 27 |
| Number of ties between paper and ChainerRL | 3 |


| Game        | ChainerRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 9672.1| N/A|
| Alien | **12484.3**| 7022|
| Amidar | 2392.3| **2946**|
| Assault | 24731.9| **29091**|
| Asterix | **454846.7**| 342016|
| Asteroids | **3885.9**| 2898|
| Atlantis | 946912.5| **978200**|
| BankHeist | 1326.3| **1416**|
| BattleZone | **69316.2**| 42244|
| BeamRider | 38111.4| **42776**|
| Berzerk | **138167.9**| 1053|
| Bowling | 84.3| **86.5**|
| Boxing | **99.9**| 99.8|
| Breakout | 658.6| **734**|
| Carnival | 5267.2| N/A|
| Centipede | 11265.2| **11561**|
| ChopperCommand | **43466.9**| 16836|
| CrazyClimber | 178111.6| **179082**|
| DemonAttack | **134637.5**| 128580|
| DoubleDunk | **8.3**| 5.6|
| Enduro | **2363.3**| 2359|
| FishingDerby | **39.3**| 33.8|
| Freeway | **34.0**| **34.0**|
| Frostbite | **8531.3**| 4342|
| Gopher | 116037.5| **118365**|
| Gravitar | **1010.8**| 911|
| Hero | 27639.9| **28386**|
| IceHockey | -0.3| **0.2**|
| Jamesbond | 27959.5| **35108**|
| JourneyEscape | -685.6| N/A|
| Kangaroo | **15517.7**| 15487|
| Krull | 9809.3| **10707**|
| KungFuMaster | **87566.3**| 73512|
| MontezumaRevenge | **0.6**| 0.0|
| MsPacman | 5786.5| **6349**|
| NameThisGame | **23151.3**| 22682|
| Phoenix | **145318.8**| 56599|
| Pitfall | 0.0| 0.0|
| Pong | **21.0**| **21.0**|
| Pooyan | 28041.5| N/A|
| PrivateEye | **289.9**| 200|
| Qbert | 24950.3| **25750**|
| Riverraid | **20716.1**| 17765|
| RoadRunner | **63523.6**| 57900|
| Robotank | **77.1**| 62.5|
| Seaquest | 27045.5| **30140**|
| Skiing | -9354.7| **-9289**|
| Solaris | 7423.3| **8007**|
| SpaceInvaders | 27810.9| **28888**|
| StarGunner | **189208.0**| 74677|
| Tennis | **23.8**| 23.6|
| TimePilot | **12758.3**| 12236|
| Tutankham | **337.4**| 293|
| UpNDown | 83140.0| **88148**|
| Venture | 289.0| **1318**|
| VideoPinball | 664013.5| **698045**|
| WizardOfWor | 20892.8| **31190**|
| YarsRevenge | **30385.0**| 28379|
| Zaxxon | 14754.4| **21772**|


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

| Training time (in days) across all domains  |               |
| --------------------------------------------|:-------------:|
| Mean |  9.21 |
| Min  | 8.52 (YarsRevenge) |
| Max  | 10.01 (Freeway) |




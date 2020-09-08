# DQN
This example trains a DQN agent, from the following paper: [Human-level control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). 

## Requirements

- atari_py>=0.1.1
- opencv-python
- filelock

## Running the Example

To run the training example:
```
python train_dqn.py [options]
```
We have already pretrained models from this script for all the domains listed in the [results](#Results). Note that while we may have run multiple seeds, our pretrained model represents a single run from this script, and may not be achieve the performance of the [results](#Results). To load a pretrained model:

```
python train_dqn.py --demo --load-pretrained --env BreakoutNoFrameskip-v4 --pretrained-type best --gpu -1
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
These results reflect PFRL commit hash: `a0ad6a7`.

 Note that the scores reported in the DQN paper are from a single run on each domain.


| Results Summary ||
| ------------- |:-------------:|
| Reporting Protocol | A re-evaluation of the best intermediate agent |
| Number of seeds | 6 |
| Number of common domains | 49 |
| Number of domains where paper scores higher | 21 |
| Number of domains where PFRL scores higher | 28 |
| Number of ties between paper and PFRL | 0 | 

 The "Original Reported Scores" are obtained from _Extended Data Table 2_ in [Human-level control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

| Game        | PFRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 6020.3| N/A|
| Alien | 1976.3| **3069**|
| Amidar | **976.6**| 739.5|
| Assault | **3542.6**| 3359|
| Asterix | 5715.3| **6012**|
| Asteroids | 1596.0| **1629**|
| Atlantis | **97512.8**| 85641|
| BankHeist | **663.2**| 429.7|
| BattleZone | 5144.4| **26300**|
| BeamRider | **7146.6**| 6846|
| Berzerk | 658.8| N/A|
| Bowling | **55.5**| 42.4|
| Boxing | **89.3**| 71.8|
| Breakout | 352.5| **401.2**|
| Carnival | 5249.2| N/A|
| Centipede | 5058.7| **8309**|
| ChopperCommand | 4737.2| **6687**|
| CrazyClimber | 103234.4| **114103**|
| DemonAttack | 9208.2| **9711**|
| DoubleDunk | **-10.9**| -18.1|
| Enduro | **307.4**| 301.8|
| FishingDerby | **14.3**| -0.8|
| Freeway | 20.6| **30.3**|
| Frostbite | **1388.4**| 328.3|
| Gopher | 7947.2| **8520**|
| Gravitar | **471.9**| 306.7|
| Hero | 19588.2| **19950**|
| IceHockey | -2.7| **-1.6**|
| Jamesbond | **765.8**| 576.7|
| JourneyEscape | -1713.3| N/A|
| Kangaroo | **8345.6**| 6740|
| Krull | **5679.2**| 3805|
| KungFuMaster | **27362.2**| 23270|
| MontezumaRevenge | **0.6**| 0.0|
| MsPacman | **2776.4**| 2311|
| NameThisGame | **7279.9**| 7257|
| Phoenix | 9406.4| N/A|
| Pitfall | -4.7| N/A|
| Pong | **20.0**| 18.9|
| Pooyan | 3446.6| N/A|
| PrivateEye | **2196.5**| 1788|
| Qbert | **10675.1**| 10596|
| Riverraid | 7554.0| **8316**|
| RoadRunner | **36572.8**| 18257|
| Robotank | 47.5| **51.6**|
| Seaquest | **6252.0**| 5286|
| Skiing | -12426.3| N/A|
| Solaris | 1396.3| N/A|
| SpaceInvaders | 1609.8| **1976**|
| StarGunner | 57293.9| **57997**|
| Tennis | **-1.8**| -2.5|
| TimePilot | 5802.8| **5947**|
| Tutankham | 148.2| **186.7**|
| UpNDown | **11110.7**| 8456|
| Venture | **517.8**| 380.0|
| VideoPinball | 14376.7| **42684**|
| WizardOfWor | 2202.2| **3393**|
| YarsRevenge | 6602.9| N/A|
| Zaxxon | **6191.1**| 4977|


## Evaluation Protocol
Our evaluation protocol is designed to mirror the evaluation protocol of the original paper as closely as possible, in order to offer a fair comparison of the quality of our example implementation. Specifically, the details of our evaluation (also can be found in the code) are the following:

- **Evaluation Frequency**: The agent is evaluated after every 1 million frames (250K timesteps). This results in a total of 200 "intermediate" evaluations.
- **Evaluation Phase**: The agent is evaluated for 500K frames (125K timesteps) in each intermediate evaluation. 
	- **Output**: The output of an intermediate evaluation phase is a score representing the mean score of all completed evaluation episodes within the 125K timesteps. If there is any unfinished episode by the time the 125K timestep evaluation phase is finished, that episode is discarded.
- **Intermediate Evaluation Episode**: 
	- We did not cap the length of the intermediate evaluation episodes.
	- Each evaluation episode begins with a random number of no-ops (up to 30), where this number is chosen uniformly at random.
	- During evaluation episodes the agent uses an epsilon-greedy policy, with epsilon=0.05.
- **Reporting**: For each run of our DQN example, we take the network weights of the best intermediate agent (i.e. the network weights that achieved the highest intermediate evaluation), and re-evaluate that agent for 30 episodes. In each of these 30 "final evaluation" episodes, the episode is terminated after 5 minutes of play (5 minutes = 300 seconds * 60 frames-per-second / 4 frames per action = 4500 timesteps). We then output the average of these 30 episodes as the achieved score for the DQN agent. The reported value in the table consists of the average of 5 "final evaluations", or 5 runs of this DQN example, each using a different random seed.


## Training times
All jobs were ran on a single GPU.

| Training time (in days) across all runs (# domains x # seeds) | |
| ------------- |:-------------:|
| Mean        |  3.613 |
| Standard deviation | 0.212|
| Fastest run | 2.767|
| Slowest run | 4.2|

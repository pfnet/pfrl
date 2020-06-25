# DQN
This example trains a DQN agent, from the following paper: [Human-level control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). 

## Requirements

- atari_py>=0.1.1
- opencv-python

## Running the Example

To run the training example:
```
python train_dqn.py [options]
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

These results reflect ChainerRL  `v0.6.0`.

The summary of the results is as follows:
 - These results are averaged over 5 runs per domain
 - We ran this example on 59 Atari domains. 
 - The original DQN paper paper ran results on 49 Atari domains. Within these 49 domains the results are as follows:
 	- On 26 domains, the scores reported in the DQN paper are higher than the scores achieved by ChainerRL.
 	- On 22 domains, our DQN implementation outscores the reported results from the DQN paper.
 	- On 1 domain, our implementation ties the reported score from the DQN paper.
 - Note that the scores reported in the DQN paper are from a single run on each domain, and might not be an accurate reflection of the DQN's true performance.
 - The "Original Reported Scores" are obtained from _Extended Data Table 2_ in [Human-level control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)


| Game        | ChainerRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 6450.5| N/A|
| Alien | 1713.1| **3069**|
| Amidar | **986.7**| 739.5|
| Assault | 3317.2| **3359**|
| Asterix | 5936.7| **6012**|
| Asteroids | 1584.5| **1629**|
| Atlantis | **96456.0**| 85641|
| BankHeist | **645.0**| 429.7|
| BattleZone | 5313.3| **26300**|
| BeamRider | **7042.9**| 6846|
| Berzerk | 707.2| N/A|
| Bowling | **52.3**| 42.4|
| Boxing | **89.6**| 71.8|
| Breakout | 364.9| **401.2**|
| Carnival | 5222.0| N/A|
| Centipede | 5112.6| **8309**|
| ChopperCommand | 6170.0| **6687**|
| CrazyClimber | 108472.7| **114103**|
| DemonAttack | 9044.3| **9711**|
| DoubleDunk | **-9.7**| -18.1|
| Enduro | 298.2| **301.8**|
| FishingDerby | **11.6**| -0.8|
| Freeway | 8.1| **30.3**|
| Frostbite | **1093.9**| 328.3|
| Gopher | 8370.0| **8520**|
| Gravitar | **445.7**| 306.7|
| Hero | **20538.7**| 19950|
| IceHockey | -2.4| **-1.6**|
| Jamesbond | **851.7**| 576.7|
| JourneyEscape | -1894.0| N/A|
| Kangaroo | **8831.3**| 6740|
| Krull | **6215.0**| 3805|
| KungFuMaster | **27616.7**| 23270|
| MontezumaRevenge | **0.0**| **0.0**|
| MsPacman | **2526.6**| 2311|
| NameThisGame | 7046.5| **7257**|
| Phoenix | 7054.4| N/A|
| Pitfall | -28.3| N/A|
| Pong | **20.1**| 18.9|
| Pooyan | 3118.7| N/A|
| PrivateEye | 1538.3| **1788**|
| Qbert | 10516.0| **10596**|
| Riverraid | 7784.1| **8316**|
| RoadRunner | **37092.0**| 18257|
| Robotank | 47.4| **51.6**|
| Seaquest | **6075.7**| 5286|
| Skiing | -13030.2| N/A|
| Solaris | 1565.1| N/A|
| SpaceInvaders | 1583.2| **1976**|
| StarGunner | 56685.3| **57997**|
| Tennis | -5.4| **-2.5**|
| TimePilot | 5738.7| **5947**|
| Tutankham | 141.9| **186.7**|
| UpNDown | **11821.5**| 8456|
| Venture | **656.7**| 380.0|
| VideoPinball | 9194.5| **42684**|
| WizardOfWor | 1957.3| **3393**|
| YarsRevenge | 4397.3| N/A|
| Zaxxon | **5698.7**| 4977|


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

We ran this DQN example 5 times for each of 59 Atari domains, for a total of 295 runs. Over these 295 runs, on average our implementation took **4.23 days** on a single GPU. Looking at the average training time for individual domains (over 5 runs for that domain), we find that YarsRevenge finishes the most quickly, taking only **3.23 days** on average. The slowest domain was UpNDown, which took **4.67 days** to complete on average.

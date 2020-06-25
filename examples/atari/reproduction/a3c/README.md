# A3C
This example trains an Asynchronous Advantage Actor Critic (A3C) agent, from the following paper: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783). 

## Requirements

- atari_py>=0.1.1
- opencv-python

## Running the Example

To run the training example:
```
python train_a3c.py [options]
```

### Useful Options
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.
- `--demo`. Runs an evaluation, instead of training the agent.
- (Currently unsupported) `--load-pretrained` Loads the pretrained model. Both `--load` and `--load-pretrained` cannot be used together.

To view the full list of options, either view the code or run the example with the `--help` option.


## Results
**NOTE: These results reflect scores from our predecessor library, ChainerRL. We will release benchmarks for PFRL in the future.**
These results reflect ChainerRL  `v0.7.0`. The reported results are compared against the scores from the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295), since the original paper does not report scores for the no-op evaluation protocol.


| Results Summary ||
| ------------- |:-------------:|
| Reporting Protocol | The highest mean intermediate evaluation score |
| Number of seeds | 1 |
| Number of common domains | 54 |
| Number of domains where paper scores higher | 25 |
| Number of domains where ChainerRL scores higher | 26 |
| Number of ties between paper and ChainerRL | 3 | 


| Game        | ChainerRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 3767.8| N/A|
| Alien | 1600.7| **2027**|
| Amidar | 873.1| **904**|
| Assault | **4819.8**| 2879|
| Asterix | **10792.4**| 6822|
| Asteroids | **2691.2**| 2544|
| Atlantis | **806650.0**| 422700|
| BankHeist | **1327.9**| 1296|
| BattleZone | 4208.8| **16411**|
| BeamRider | 8946.9| **9214**|
| Berzerk | **1527.2**| 1022|
| Bowling | 31.7| **37**|
| Boxing | **99.0**| 91|
| Breakout | **575.9**| 496|
| Carnival | 5121.9| N/A|
| Centipede | **5647.5**| 5350|
| ChopperCommand | **5916.3**| 5285|
| CrazyClimber | 120583.3| **134783**|
| Defender | N/A| 52917.0|
| DemonAttack | **112456.3**| 37085|
| DoubleDunk | 1.5| **3**|
| Enduro | **0.0**| **0**|
| FishingDerby | **37.7**| -7|
| Freeway | **0.0**| **0**|
| Frostbite | **312.6**| 288|
| Gopher | **10608.9**| 7992|
| Gravitar | 250.5| **379**|
| Hero | **36264.3**| 30791|
| IceHockey | -4.5| **-2**|
| Jamesbond | 373.7| **509**|
| JourneyEscape | -1026.5| N/A|
| Kangaroo | 107.0| **1166**|
| Krull | 9260.2| **9422**|
| KungFuMaster | **37750.0**| 37422|
| MontezumaRevenge | 2.6| **14**|
| MsPacman | **2851.4**| 2436|
| NameThisGame | **11301.1**| 7168|
| Phoenix | **38671.4**| 9476|
| Pitfall | -2.0| **0**|
| Pong | **20.9**| 7|
| Pooyan | 4328.9| N/A|
| PrivateEye | 725.3| **3781**|
| Qbert | **19831.0**| 18586|
| Riverraid | 13172.8| N/A|
| RoadRunner | 40348.1| **45315**|
| Robotank | 3.0| **6**|
| Seaquest | **1789.5**| 1744|
| Skiing | -15820.1| **-12972**|
| Solaris | 3395.6| **12380**|
| SpaceInvaders | **1739.5**| 1034|
| StarGunner | **60591.7**| 49156|
| Surround | N/A| -8.0|
| Tennis | -13.1| **-6**|
| TimePilot | 4077.5| **10294**|
| Tutankham | **274.5**| 213|
| UpNDown | 78790.0| **89067**|
| Venture | **0.0**| **0**|
| VideoPinball | **518840.8**| 229402|
| WizardOfWor | 2488.4| **8953**|
| YarsRevenge | 14217.7| **21596**|
| Zaxxon | 86.8| **16544**|


## Evaluation Protocol

Our evaluation protocol is designed to mirror the evaluation protocol from the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295) as closely as possible, since the original A3C paper does not report reproducible results (they use human starts trajectories which are not publicly available). The reported results are from the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295), Table 3.

Our evaluation protocol is designed to mirror the evaluation protocol of the original paper as closely as possible, in order to offer a fair comparison of the quality of our example implementation. Specifically, the details of our evaluation (also can be found in the code) are the following:

- **Evaluation Frequency**: The agent is evaluated after every 1 million frames (250K timesteps). This results in a total of 200 "intermediate" evaluations.
- **Evaluation Phase**: The agent is evaluated for 500K frames (125K timesteps) in each intermediate evaluation. 
	- **Output**: The output of an intermediate evaluation phase is a score representing the mean score of all completed evaluation episodes within the 125K timesteps. If there is any unfinished episode by the time the 125K timestep evaluation phase is finished, that episode is discarded.
- **Intermediate Evaluation Episode**: 
	- Each intermediate evaluation episode is capped in length at 27K timesteps or 108K frames.
	- Each evaluation episode begins with a random number of no-ops (up to 30), where this number is chosen uniformly at random.
- **Reporting**: For each run of our A3C example, we report the highest scores amongst each of the intermediate evaluation phases. This differs from the original A3C paper which states that: "We additionally used the final network weights for evaluation". This is because the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295) states that "Per-game maximum scores are computed by taking the maximum raw scores of the agent and then averaging over three seeds".


## Training times

We trained with 17 CPUs and no GPU. However, we used 16 processes (as per the A3C paper).


| Training time (in days) across all domains | |
| ------------- |:-------------:|
| Mean        |  1.158 |
| Fastest Domain |1.008 (Asteroids)|
| Slowest Domain | 1.46 (ChopperCommand)|

				

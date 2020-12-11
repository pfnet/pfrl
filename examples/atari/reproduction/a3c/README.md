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

We have already trained models from this script for all the domains listed in the [results](#Results). To load a pretrained model:
```
python train_a3c.py --demo --load-pretrained --env BreakoutNoFrameskip-v4 --pretrained-type best
```

### Useful Options
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.
- `--demo`. Runs an evaluation, instead of training the agent.
- `--load-pretrained` Loads the pretrained model. Both `--load` and `--load-pretrained` cannot be used together.
- `--pretrained-type`. Either `best` (the best intermediate network during training) or `final` (the final network after training).

To view the full list of options, either view the code or run the example with the `--help` option.


## Results
These results reflect PFRL commit hash: `39918e2`. The reported results are compared against the scores from the [Noisy Networks Paper](https://arxiv.org/abs/1706.10295), since the original paper does not report scores for the no-op evaluation protocol.


| Results Summary ||
| ------------- |:-------------:|
| Reporting Protocol | The highest mean intermediate evaluation score |
| Number of seeds | 3 |
| Number of common domains | 55 |
| Number of domains where paper scores higher | 25 |
| Number of domains where PFRL scores higher | 28 |
| Number of ties between paper and PFRL | 2 | 


| Game        | PFRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| Adventure | -0.1| N/A|
| AirRaid | 6361.9| N/A|
| Alien | 1809.6| **2027**|
| Amidar | 834.5| **904**|
| Assault | **7035.0**| 2879|
| Asterix | **12577.4**| 6822|
| Asteroids | **2703.2**| 2544|
| Atlantis | **874883.3**| 422700|
| BankHeist | **1323.0**| 1296|
| BattleZone | 10514.6| **16411**|
| BeamRider | 8882.6| **9214**|
| Berzerk | 877.5| **1022**|
| Bowling | 31.3| **37**|
| Boxing | **97.5**| 91|
| Breakout | **581.0**| 496|
| Carnival | 5517.9| N/A|
| Centipede | 4837.8| **5350**|
| ChopperCommand | **6001.3**| 5285|
| CrazyClimber | 119886.3| **134783**|
| Defender | **860555.6**| 52917|
| DemonAttack | **106654.2**| 37085|
| DoubleDunk | 1.5| **3**|
| ElevatorAction | 45596.7| N/A|
| Enduro | **0.0**| **0**|
| FishingDerby | **40.9**| -7|
| Freeway | **0.0**| **0**|
| Frostbite | **295.6**| 288|
| Gopher | **8154.3**| 7992|
| Gravitar | 248.2| **379**|
| Hero | 24205.5| **30791**|
| IceHockey | -5.1| **-2**|
| Jamesbond | 285.7| **509**|
| JourneyEscape | -968.2| N/A|
| Kangaroo | 63.9| **1166**|
| Krull | **10028.7**| 9422|
| KungFuMaster | **39291.6**| 37422|
| MontezumaRevenge | 2.2| **14**|
| MsPacman | **2808.1**| 2436|
| NameThisGame | **9053.6**| 7168|
| Phoenix | **42386.3**| 9476|
| Pitfall | -2.7| **0**|
| Pong | **20.9**| 7|
| Pooyan | 4214.5| N/A|
| PrivateEye | 370.8| **3781**|
| Qbert | **20721.9**| 18586|
| Riverraid | 13577.5| N/A|
| RoadRunner | 37228.9| **45315**|
| Robotank | 3.0| **6**|
| Seaquest | **1781.9**| 1744|
| Skiing | **-11275.6**| -12972|
| Solaris | 3795.4| **12380**|
| SpaceInvaders | **1043.5**| 1034|
| StarGunner | **55485.9**| 49156|
| Surround | N/A| -8|
| Tennis | -6.8| **-6**|
| TimePilot | 5253.8| **10294**|
| Tutankham | **324.1**| 213|
| UpNDown | 60758.2| **89067**|
| Venture | **1.2**| 0|
| VideoPinball | **284500.2**| 229402|
| WizardOfWor | 3056.6| **8953**|
| YarsRevenge | **22862.5**| 21596|
| Zaxxon | 67.3| **16544**|



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

We trained with 16 CPUs and no GPU.


| Training time (in hours) across all runs (# domains x # seeds) | |
| ------------- |:-------------:|
| Mean        |  12.66 |
| Standard deviation | 0.876|
| Fastest run | 10.968|
| Slowest run | 15.212|


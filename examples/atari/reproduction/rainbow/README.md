# Rainbow
This example trains a Rainbow agent, from the following paper: [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298). 

## Requirements

- atari_py>=0.1.1
- opencv-python

## Running the Example

To run the training example:
```
python train_rainbow.py [options]
```

We have already pretrained models from this script for all the domains listed in the [results](#Results) section. To load a pretrained model:

```
python train_rainbow.py --demo --load-pretrained --env BreakoutNoFrameskip-v4 --pretrained-type best --gpu -1
```

### Useful Options
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_rainbow.py --gpu -1`.
- `--env`. Specifies the environment. 
- `--render`. Add this option to render the states in a GUI window.
- `--seed`. This option specifies the random seed used.
- `--outdir` This option specifies the output directory to which the results are written.
- `--demo`. Runs an evaluation, instead of training the agent.
- `--load-pretrained` Loads the pretrained model. Both `--load` and `--load-pretrained` cannot be used together.
- `--pretrained-type`. Either `best` (the best intermediate network during training) or `final` (the final network after training).

To view the full list of options, either view the code or run the example with the `--help` option.


## Results
These results reflect PFRL commit hash:  `a0ad6a7`.

| Results Summary ||
| ------------- |:-------------:|
| Reporting Protocol | A re-evaluation of the best intermediate agent |
| Number of seeds | 2 |
| Number of common domains | 52 |
| Number of domains where paper scores higher | 21 |
| Number of domains where PFRL scores higher | 31 |
| Number of ties between paper and PFRL | 0 | 


| Game        | PFRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 6616.1| N/A|
| Alien | **10255.4**| 9491.7|
| Amidar | 4284.6| **5131.2**|
| Assault | **15331.9**| 14198.5|
| Asterix | **550307.6**| 428200.3|
| Asteroids | **3399.6**| 2712.8|
| Atlantis | **883073.0**| 826659.5|
| BankHeist | 1272.7| **1358.0**|
| BattleZone | **202382.5**| 62010.0|
| BeamRider | **21661.5**| 16850.2|
| Berzerk | **6018.1**| 2545.6|
| Bowling | **62.3**| 30.0|
| Boxing | **99.9**| 99.6|
| Breakout | 317.8| **417.5**|
| Carnival | 5687.1| N/A|
| Centipede | 7546.8| **8167.3**|
| ChopperCommand | **21014.5**| 16654.0|
| CrazyClimber | **174025.0**| 168788.5|
| Defender | N/A| 55105.0|
| DemonAttack | 100980.6| **111185.2**|
| DoubleDunk | **-0.1**| -0.3|
| Enduro | **2281.9**| 2125.9|
| FishingDerby | **39.1**| 31.3|
| Freeway | 33.6| **34.0**|
| Frostbite | **11046.3**| 9590.5|
| Gopher | **76872.9**| 70354.6|
| Gravitar | 1387.2| **1419.3**|
| Hero | 34234.5| **55887.4**|
| IceHockey | **6.6**| 1.1|
| Jamesbond | 23222.9| N/A|
| JourneyEscape | -184.8| N/A|
| Kangaroo | 13726.5| **14637.5**|
| Krull | 7844.4| **8741.5**|
| KungFuMaster | **54835.5**| 52181.0|
| MontezumaRevenge | 21.0| **384.0**|
| MsPacman | 5277.3| **5380.4**|
| NameThisGame | **14679.0**| 13136.0|
| Phoenix | **147467.9**| 108528.6|
| Pitfall | -3.4| **0.0**|
| Pong | **21.0**| 20.9|
| Pooyan | 15040.2| N/A|
| PrivateEye | 101.7| **4234.0**|
| Qbert | **42518.7**| 33817.5|
| Riverraid | 30121.2| N/A|
| RoadRunner | **67638.5**| 62041.0|
| Robotank | **74.0**| 61.4|
| Seaquest | 5277.4| **15898.9**|
| Skiing | -29974.7| **-12957.8**|
| Solaris | **6730.1**| 3560.3|
| SpaceInvaders | 2823.1| **18789.0**|
| StarGunner | **155248.8**| 127029.0|
| Surround | N/A| 9.7|
| Tennis | -0.1| **0.0**|
| TimePilot | **24038.2**| 12926.0|
| Tutankham | **258.0**| 241.0|
| UpNDown | 258505.5| N/A|
| Venture | 2.5| **5.5**|
| VideoPinball | 292835.7| **533936.5**|
| WizardOfWor | **21341.2**| 17862.5|
| YarsRevenge | 93877.2| **102557.0**|
| Zaxxon | **25084.0**| 22209.5|



## Evaluation Protocol
Our evaluation protocol is designed to mirror the evaluation protocol of the original paper as closely as possible, in order to offer a fair comparison of the quality of our example implementation. Specifically, the details of our evaluation (also can be found in the code) are the following:

- **Evaluation Frequency**: The agent is evaluated after every 1 million frames (250K timesteps). This results in a total of 200 "intermediate" evaluations.
- **Evaluation Phase**: The agent is evaluated for 500K frames (125K timesteps) in each intermediate evaluation. 
	- **Output**: The output of an intermediate evaluation phase is a score representing the mean score of all completed evaluation episodes within the 125K timesteps. If there is any unfinished episode by the time the 125K timestep evaluation phase is finished, that episode is discarded.
- **Intermediate Evaluation Episode**: 
	- Capped at 30 mins of play, or 108K frames/ 27K timesteps.
	- Each evaluation episode begins with a random number of no-ops (up to 30), where this number is chosen uniformly at random.
	- During evaluation episodes the agent uses an epsilon-greedy policy, with epsilon=0.001 (original paper does greedy evaluation because noisy networks are used)
- **Reporting**: For each run of our Rainbow example, we take the network weights of the best intermediate agent (i.e. the network weights that achieved the highest intermediate evaluation), and re-evaluate that agent for 200 episodes. In each of these 200 "final evaluation" episodes, the episode is terminated after 30 minutes of play (30 minutes = 1800 seconds * 60 frames-per-second / 4 frames per action = 27000 timesteps). We then output the average of these 200 episodes as the achieved score for the Rainbow agent. The reported value in the table consists of the average of 1 "final evaluation", or 1 run of this Rainbow example.


## Training times
All jobs were run on a single GPU.

| Training time (in days) across all runs (# domains x # seeds) | |
| ------------- |:-------------:|
| Mean        |  7.173 |
| Standard deviation | 0.166|
| Fastest run | 6.764|
| Slowest run | 7.612|




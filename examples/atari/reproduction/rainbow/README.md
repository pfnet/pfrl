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

### Useful Options
- `--gpu`. Specifies the GPU. If you do not have a GPU on your machine, run the example with the option `--gpu -1`. E.g. `python train_rainbow.py --gpu -1`.
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

These results reflect ChainerRL  `v0.7.0`.

| Results Summary ||
| ------------- |:-------------:|
| Reporting Protocol | A re-evaluation of the best intermediate agent |
| Number of seeds | 1 |
| Number of common domains | 52 |
| Number of domains where paper scores higher | 20 |
| Number of domains where ChainerRL scores higher | 30 |
| Number of ties between paper and ChainerRL | 2 | 


| Game        | ChainerRL Score           | Original Reported Scores |
| ------------- |:-------------:|:-------------:|
| AirRaid | 6500.9| N/A|
| Alien | 9409.1| **9491.7**|
| Amidar | 3252.7| **5131.2**|
| Assault | **15245.5**| 14198.5|
| Asterix | 353258.5| **428200.3**|
| Asteroids | **2792.3**| 2712.8|
| Atlantis | **894708.5**| 826659.5|
| BankHeist | **1734.8**| 1358.0|
| BattleZone | **90625.0**| 62010.0|
| BeamRider | **27959.5**| 16850.2|
| Berzerk | **26704.2**| 2545.6|
| Bowling | **67.1**| 30.0|
| Boxing | **99.8**| 99.6|
| Breakout | 340.8| **417.5**|
| Carnival | 5530.3| N/A|
| Centipede | 7718.1| **8167.3**|
| ChopperCommand | **303480.5**| 16654.0|
| CrazyClimber | 165370.0| **168788.5**|
| Defender | N/A| 55105.0|
| DemonAttack | 110028.0| **111185.2**|
| DoubleDunk | **-0.1**| -0.3|
| Enduro | **2273.8**| 2125.9|
| FishingDerby | **45.3**| 31.3|
| Freeway | 33.7| **34.0**|
| Frostbite | **10432.3**| 9590.5|
| Gopher | **76662.9**| 70354.6|
| Gravitar | **1819.5**| 1419.3|
| Hero | 12590.5| **55887.4**|
| IceHockey | **5.1**| 1.1|
| Jamesbond | 31392.0| N/A|
| JourneyEscape | 0.0| N/A|
| Kangaroo | 14462.5| **14637.5**|
| Krull | 7989.0| **8741.5**|
| KungFuMaster | 22820.5| **52181.0**|
| MontezumaRevenge | 4.0| **384.0**|
| MsPacman | **6153.4**| 5380.4|
| NameThisGame | **14035.1**| 13136.0|
| Phoenix | 5169.6| **108528.6**|
| Pitfall | **0.0**| **0.0**|
| Pong | **20.9**| **20.9**|
| Pooyan | 7793.1| N/A|
| PrivateEye | 100.0| **4234.0**|
| Qbert | **42481.1**| 33817.5|
| Riverraid | 26114.0| N/A|
| RoadRunner | **64306.0**| 62041.0|
| Robotank | **74.4**| 61.4|
| Seaquest | 4286.8| **15898.9**|
| Skiing | **-9441.0**| -12957.8|
| Solaris | **7902.2**| 3560.3|
| SpaceInvaders | 2838.0| **18789.0**|
| StarGunner | **181192.5**| 127029.0|
| Surround | N/A| 9.7|
| Tennis | -0.1| **0.0**|
| TimePilot | **25582.0**| 12926.0|
| Tutankham | **251.9**| 241.0|
| UpNDown | 284465.6| N/A|
| Venture | **1499.0**| 5.5|
| VideoPinball | 492071.8| **533936.5**|
| WizardOfWor | **19796.5**| 17862.5|
| YarsRevenge | 80817.2| **102557.0**|
| Zaxxon | **26827.5**| 22209.5|




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

Time statistics...

| Training time (in days) across all domains | |
| ------------- |:-------------:|
| Mean        |  12.929 |
| Fastest Domain |11.931 (Frostbite)|
| Slowest Domain | 13.974 (UpNDown)|




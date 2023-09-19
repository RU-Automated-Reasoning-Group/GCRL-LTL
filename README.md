# Instructing goal-conditioned agents with LTL objectives

Goal-conditioned reinforcement learning (RL) is a powerful approach for learning general-purpose skills by reaching diverse goals. However, it has limitations when it comes to task-conditioned policies, where goals are specified by temporally extended instructions written in the Linear Temporal Logic (LTL) formal language. Existing approaches for finding LTL-satisfying policies rely on sampling a large set of LTL instructions during training to adapt to unseen tasks at inference time. However, these approaches do not guarantee generalization to out-of-distribution LTL objectives, which may have increased complexity. In this work, we developed a novel neurosymbolic approach to address this challenge. We showed that simple goal-conditioned RL agents can be instructed to follow arbitrary LTL specifications without additional training over the LTL task space. 

## ZoneEnv

We use a robot from Safety Gym called Point, with one actuator for turning and another for moving forward or backward. An agent can observe the LiDAR information of its surrounding zones. Given this indirect geographical information, it has to visit and/or avoid certain zones to satisfy sampled LTL task specifications. The initial positions of the zones and the robot are random in every episode.

### Setup
* Using conda and install `pygraphviz`.
    ```bash
    conda install -c conda-forge pygraphviz
    ```
* Install [mujoco](https://www.roboti.us) and [mujoco-py](https://github.com/openai/mujoco-py).
* Install [safty-gym](https://github.com/openai/safety-gym).
* Install required pip packages
    ```
    numpy
    torch
    stable-baslines3
    graphviz
    gym
    mujoco-py
    ```

### Training (optional)
* Training primitive action policies for `ZoneEnv`, including `UP`, `DOWN`, `LEFT`, and `RIGHT`:
    ```
    python train_primitives.py
    ```
* Training goal-conditioned policy for `ZoneEnv`:
    ```bash
    python train_gc_policy.py
    ```
### Models
* Primitive action policies for navigating the `Point` robot are saved in:
    ```
    [project_base]/zones/models/primitives/*.zip
    ```
* Trained goal-conditioned policies are saved in:
    ```
    [project_base]/zones/models/goal-conditioned/best_model_ppo_[N].zip
    ```
    where `N` denotes the number of zones present in the environment (8 by default).

### Experiments
* Avoidance experiments e.g. $\neg y U (j \wedge (\neg wUr))$ (where $y$ for yellow, $j$ for jet-black, $w$ for white, and $r$ for red).
    ```bash
    python exp.py --task='avoid'
    ```
* Loop experiments e.g. $GF(r \wedge XF y) \wedge G(\neg w)$
    ```bash
    python exp.py --task='traverse'
    ```
* Goal-chaining experiments e.g. $F(j \wedge F(w \wedge F(r \wedge Fy)))$
    ```bash
    python exp.py --task='chain'
    ```
* Stability experiments e.g $FGy$
    ```bash
    python exp.py --task='stable'
    ```
    See script `[project_base]/zones/exp.py` for more details including specifying `eval_repeats` and `device`, etc.

### Examples
* The left and right figures show the trajectory for the task $\neg y U (j \wedge (\neg wUr))$.
<figure>
<p align='center'>
<img src='./zones/assets/avoid_1.gif'  height=300 width=300>
<img src='./zones/assets/avoid_2.gif'  height=300 width=300>
<center>
<hr><br>
</figure>

* The left and right figures show the trajectory for the task $F(j \wedge X(\neg y U r)) \wedge G(\neg w) $.
<figure>
<p align='center'>
    <img src='./zones/assets/avoid_more_1.gif'  height=300 width=300>
    <img src='./zones/assets/avoid_more_2.gif'  height=300 width=300>
</p>
<center>
<hr><br>
</figure>

* The left and right figures show the trajectories for the task $F(j \wedge F(w \wedge F(r \wedge Fy)))$.
<figure>
<p align='center'>
    <img src='./zones/assets/chain_1.gif'  height=300 width=300>
    <img src='./zones/assets/chain_2.gif'  height=300 width=300>
</p>
<hr><br>
</figure>

* The left and right figures show the trajectories for the task $GF(r \wedge XF y) \wedge G(\neg w)$.
<figure>
<p align='center'>
    <img src='./zones/assets/traverse_1.gif'  height=300 width=300>
    <img src='./zones/assets/traverse_2.gif'  height=300 width=300>
</p>
<hr><br>
</figure>

* The left and right figures show the trajectories for the task $FGy$.
<figure>
<p align='center'>
    <img src='./zones/assets/stable_1.gif'  height=300 width=300>
    <img src='./zones/assets/stable_2.gif'  height=300 width=300>
</p>
<hr><br>
</figure>

* The left figure shows the trajectory for the task $F(j \wedge r)$.
* The right figure shows the trajectory for the task $F(j \wedge \neg r)$.
<figure>
<p align='center'>
    <img src='./zones/assets/intersect.gif'  height=300 width=300>
    <img src='./zones/assets/bypass.gif'  height=300 width=300>
</p>
<hr><br>
</figure>

* The left figure shows the trajectory for the task $GFw \wedge GFy$.
* The right figure shows the trajectory for the task $GFw \wedge GFy \wedge G(\neg j)$
<figure>
<p align='center'>
    <img src='./zones/assets/round.gif'  height=300 width=300>
    <img src='./zones/assets/round_avoid.gif'  height=300 width=300>
</p>
<hr><br>
</figure>

* The figure shows the trajectory for the task $Fj \wedge (\neg r \wedge  \neg y \wedge \neg w)Uj$
<figure>
<p align='center'>
    <img src='./zones/assets/avoid_all.gif'  height=300 width=300>
</p>
<hr><br>
</figure>

### References
* pygraphviz, https://pygraphviz.github.io
* LTL2Action, https://github.com/LTL2Action/LTL2Action
* gltl2ba, https://github.com/PatrickTrentin88/gltl2ba
* safety-gym, https://github.com/openai/safety-gym

## Ant 16 rooms

Ant-16rooms is an environment with continuous observation and action space. In this walled environment with 16 rooms, each room has the same size 8 × 8 divided by walls and corridors with a thickness of 1. There are two obstacles denoted by black squares in the environment. We place a Mujoco Ant robot in this environment for navigation.

## Setup

The environment for the Ant16rooms experiment is based on the following version of the packages:
```
	numpy=1.18.5
	torch=1.5.1
	gym=0.13.1
	mujoco_py=2.0.2.5
```
along with MuJoCo simulator version `mujoco200` from [MuJoCo release website](https://www.roboti.us/download.html)

1. Docker
	The environment is designed based on the environment used in [GCSL](https://github.com/dibyaghosh/gcsl). To download the docker image:

	```
	docker pull dibyaghosh/gcsl:0.1
	```
2. Conda
	For conda environment setting-up, please refer to [conda_environment.yml](.\ant\environment\conda_environment.yml) for all specific versions of packages.
3. Python(pip)
	For Python pip packages, please refer to [python_requirement.txt](.\ant\environment\python_requirement.txt) for all specific versions of packages.


## Run

Add workspace directory to PYTHONPATH:
```
export PYTHONPATH="${PYTHONPATH}:{path_of_GCRL-LTL_ant_folder}"
```

##### Testing with LTL specifications

```
python RRT_star/Testing_LTLSpecs_with_graph_ant16rooms.py ant16rooms {#ofspecification}
```
specifications $\phi_1$ to $\phi_5$ are corresponding to # 9 to 13 as input

## Results

#### Specification $\phi_1$

$F((0, 2) \vee (2, 0))$ - either reaching room (2,0) [orange position on the left] or room (0,2) [orange position on the right]

<figure>
<p align='center'>
  <img src="./ant/misc/fig/phi1maze.png" alt="phi1maze" height=250 width=250>
  <img src="./ant/misc/gif/phi1-cut.gif" alt="phi1" height=250 width=250>
</p>
</figure>


#### Specification $\phi_2$

$F(((0, 2) \vee (2, 0)) \wedge F(2, 2))$ - reaching room (2,2) [orange] by choosing any of the two orange paths

<p align='center'>
  <img src="./ant/misc/fig/phi2maze.png" alt="phi2maze" height=250 width=250>
  <img src="./ant/misc/gif/phi2-cut.gif" alt="phi2" height=250 width=250>
</p>

#### Specification $\phi_3$

$F(((0, 2) \vee (2, 0)) \wedge F((2, 2) \wedge F(((2, 1) \vee (3, 2)) \wedge F(3, 1))))$ - reaching room (3,1) [yellow] after visiting orange.

<p align='center'>
  <img src="./ant/misc/fig/phi3maze.png" alt="phi3maze" height=250 width=250>
  <img src="./ant/misc/gif/phi3-cut.gif" alt="phi3" height=250 width=250>
</p>

#### Specification $\phi_4$

$F(((0, 2) \vee (2, 0)) \wedge F((2, 2) \wedge F(((2, 1) \vee (3, 2)) \wedge F((3, 1) \wedge F(((1, 1) \vee (3, 3)) \wedge F(1, 3))))))$ - reaching room (1,3) [green] after visiting orange and yellow sequentially.

<p align='center'>
  <img src="./ant/misc/fig/phi4maze.png" alt="phi4maze" height=250 width=250>
  <img src="./ant/misc/gif/phi4-cut.gif" alt="phi4" height=250 width=250>
</p>

#### Specification $\phi_5$

$F(((0, 2) \vee (2, 0)) \wedge F((2, 2) \wedge F(((2, 1) \vee (3, 2)) \wedge F((3, 1) \wedge F(((1, 1) \vee (3, 3)) \wedge F((1, 3) \wedge F(((1, 1) \vee (0, 3)) \wedge F(0, 1))))))))$ - reaching room (0,1) [purple] after visiting orange, yellow and green sequentially.

<p align='center'>
  <img src="./ant/misc/fig/phi5maze.png" alt="phi5maze" height=250 width=250>
  <img src="./ant/misc/gif/phi5-cut.gif" alt="phi5" height=250 width=250>
</p>

#### $\omega$-Regular Specification $\phi_6$

$\varphi_1 \vee \varphi_2$ where $\varphi_1$ (the green path) is $GF((1, 0) ∧ X(F((3, 0) ∧ X(F(3, 2) ∧ XF(1, 2)))))$ and $\varphi_2$ (the orange path) is $F(0, 2) \wedge XGF((2, 2) \wedge X(F((3, 2) \wedge X(F(3, 3) \wedge XF(2, 3)))))$ - the agent opts to iteratively traverse a small loop to satisfy the $\omega$-regular specification, although this loop is on a far-away end.

<p align='center'>
<img src="./ant/misc/fig/phi9maze.png" alt="phi9maze" height=250 width=250>
  <img src="./ant/misc/gif/phi_loop-cut.gif" alt="phi_loop" height=250 width=250>
</p>

### 0

- ''The Most Common Habits from more than 200 English Papers written by Graduate Chinese Engineering Students  ''

​		--- Felicia Brittman

- ''The Element of Style''

### 1

1. policy learning
2. reinforcement learning
3. [imitation learning](https://smartlabai.medium.com/a-brief-overview-of-imitation-learning-8a8a75c44a9c)
   - Behavioral cloning [supervised learning]
   - (Iterative) Direct policy learning (via interactive demonstrator) [supervised learning]
   - Inverse Reinforcement Learning [RL]
4. supervised learning

### 2 goal-conditioned supervised learning (GCSL) 

**supervised imitation learning**

**no need of reward function**,  but use maximum likelihood

1. **goal-conditioned policy**
   - that seeks to obtain the indicator reward of having the observation exactly match the goal. Such a reward does not require any additional instrumentation of the environment beyond the sensors the robot already has.
2. behavioral cloning: perform supervised learning(MLE) on dataset of expert behaviors [ learn policy ***/pie*** to predict ***action*** given input *state*]

   - dataset of optimal trajectories -> imitate trajectory with **supervised learning**.

   - "Learning to Reach Goals via Iterated Supervised Learning" move it into **RL** by learning from feedback not from human expert. In **RL**, they can not imitate trajectory because these trajectory are got from a suboptimal policy, while in supervised learning, the trajectory can be used because it is  human-defined optimal. So, they need to make trajectory optimal in RL and then imitate it.

   - however, it is almost impossible to do this for all the problem, but they showed that it can be done or goal reaching problem.
3. Hindsight relabeling?
   - 
4. goal-conditioned imitation learning primitives?

   - [Goal conditioned imitation learning](https://proceedings.neurips.cc/paper/2019/file/c8d3a760ebab631565f8509d84b3b3f1-Paper.pdf) (NIPS 2019):
     - goalGAIL ([GAIL](https://proceedings.neurips.cc/paper/2016/file/cc7e2b878868cbae992d1fb743995d8f-Paper.pdf))
     -  In this work we propose a novel algorithm goalGAIL, which incorporates demonstrations to drastically speed up the convergence to a policy able to reach any goal. goalGAIL leverages the recently shown compatibility of GAIL with off-policy algorithms.

   - [Learning actionable representations with goal conditioned policies](https://proceedings.neurips.cc/paper/2019/file/c8cc6e90ccbff44c9cee23611711cdc4-Paper.pdf) (ICLR 2019)
5. Hindsight Experience Replay?
6. **off-policy and on-policy algorithm:** 

   - [On-Policy v/s Off-Policy Learning](https://towardsdatascience.com/on-policy-v-s-off-policy-learning-75089916bc2f)

   - An off-policy learner learns the value of the optimal policy independently of the agent's actions. Q-learning is an off-policy learner. An on-policy learner learns the value of the policy being carried out by the agent including the exploration steps.

   - The distinction disappears if the current policy is a greedy policy. However, such an agent would not be good since it never explores.

   - What is "**off-policy <u>data</u>**"?
7. expert’s **demonstrations** = trajectories
8. value function estimation?

   - maybe used in self-
9. self-imitation?
10. "we show that GCSL learns efficiently while training on every previous trajectory without reweighting, thereby maximizing data reuse." **WHY?**
11. Hindsight Experience Replay (HER)?
12. What does "the lower bound on a RL objective" means?
13. what is target goal distribution $p(g)$?

### 3 RETHINKING GOAL-CONDITIONED SUPERVISED LEARNING AND ITS CONNECTION TO OFFLINE RL (WGCSL) 

1. Propose: GCSL has a major disadvantage for offline goalconditioned RL, i.e., it only considers the last step reward r($s_T$, $a_T$, g) and generally results in
   suboptimal policies.
2. 

# RT,RRT,RRT*

### RT

### RRT

- efficiently search [nonconvex](https://en.wikipedia.org/wiki/Convex_space), high-dimensional spaces by randomly building a [space-filling tree](https://en.wikipedia.org/wiki/Space-filling_tree).

- one of big wins: more efficient at exploring places that have lots of obstacles, while PRM often spend most of its time doing collision detecting. [LaValle's paper]
- once you get a path from Start to Goal, any more search round will not change the length of the path. But in RRT*, it may be updated to a shorter path.

### RRT*

- forward projection (reminding: PRM)
- [alg pseudocode](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/SamplingBasedMotionPlanning.pdf) page35



# Goal

### programmatic reinforcement learning

neuro-symbolic reinforcement learning

RRT*+GCSL+4 primitives(start from it)

relevant: 

don't care about the low level action, focus on high level action using RRT.

My main idea: use RRT* to generate better trajectories when given a goal.



1. (base line) run GCSL, in env in wenjie paperc
2. RRT* + GCSL



# 4

linux GCSL:

​	ssh  xxx@sss.cs.rutgers.edu

​	ssh  wm300@sss.cs.rutgers.edu


```shell
"on sss server"
export PYTHONPATH="${PYTHONPATH}:/common/home/wm300/gcsl_ant"
python RRT_star/Generate_graph.py ant16rooms
python RRT_star/SupervisedLearning_with_graph.py ant16rooms 1000 500
```

```shell
"in docker (sss server)"
wm300@sss:~$ docker docker run -it --gpus=all dibyaghosh/gcsl:0.1 bash
*** docker run -it --gpus=all gcsl bash
[docker run -idt --gpus=all gcsll bash
docker exec -it  485b5308f6d6 bash]


export PYTHONPATH="${PYTHONPATH}:/root/code/gcsl_ant"
python experiments/gcsl_example.py ['antfall', 'antumaze', 'antpush', 'antfourrooms', 'antlongumaze']
python experiments/antenv_test.py ['antfall', 'antumaze', 'antpush', 'antfourrooms', 'antlongumaze']
python RRT_star/example.py antumaze
	[python RRT_star/Generate_graph.py ant9roomscloseddoors]
	[python RRT_star/Generate_graph.py ant16roomscloseddoors]

	
[python RRT_star/Finetuning_with_graph.py ant9roomscloseddoors]
	[python RRT_star/finetuned_graph_result_testing.py ant16rooms 18 0]

python RRT_star/SL_rrts_tree.py antumaze 1000 50/200/100(new)
	[python RRT_star/SupervisedLearning_with_graph.py antpush 200 50/200/100(new)]
	[python RRT_star/SupervisedLearning_with_graph.py ant9roomscloseddoors 400 200]
   [python RRT_star/SupervisedLearning_with_graph.py ant16rooms 1000 500]

python RRT_star/FinetuneAndSL_with_graph.py ant16rooms 1000 400 1000

python RRT_star/RRT_star_test.py antumaze
	[python RRT_star/Testing_with_graph.py ant16rooms]
	[python RRT_star/Testing_LTLSpecs_with_graph_ant9rooms.py ant9rooms]
	[python RRT_star/Testing_LTLSpecs_with_graph_ant16rooms.py ant16rooms 10]

[python RRT_star/case_test_using_graph.py antpush]

```

<del>python RRT_star/rrt_tree_only_test.py antumaze</del>

```shell
docker image tag gcsl_test wensenmao300/gcsl:test
docker push wensenmao300/gcsl:test
```

torch_version: 1.1.0 [pip install --upgrade torch torchvision ->>1.5.0]

pip install --upgrade scipy

git commit -am "update from PC“

​	 

git reset --hard f84cbacb3a1a655856ac1d93be92123d3c382702

[How to continue a Docker container which has exited](https://stackoverflow.com/questions/21928691/how-to-continue-a-docker-container-which-has-exited)

# 5. cmds

windows10 ubuntu：

anaconda3： /root/anaconda3

conda config --set auto_activate_base false

​	/mnt/c/Users/52744/OneDrive/桌面

​	/mnt/c/Users/52744/OneDrive/桌面/pi-PRL/pi-PRL-main

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro150/bin
```

python3 pi_HPRL.py -e 2

python utils/visualize_policy.py --env_name mjrl_spec_ant_fall-v1 --policy /mnt/c/Users/52744/OneDrive/桌面/pi-PRL/pi-PRL-main/data/ant_fall_3-[123]/iterations/best_policy.pickle --mode evaluation --episodes 20

python3 mjrl/utils/visualize_policy.py --env_name mjrl_spec_ant_fall-v1 --policy /mnt/c/Users/52744/OneDrive/桌面/pi-PRL/pi-PRL-main/data/ant_fall_3-[123]/iterations/best_policy.pickle --mode evaluation --episodes 20





#### Recommended approach for saving a model

There are two main approaches for serializing and restoring a model.

The first (recommended) saves and loads only the model parameters:

```py
torch.save(the_model.state_dict(), PATH)
```

Then later:

```py
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
```

The second saves and loads the entire model:

```py
torch.save(the_model, PATH)
```

Then later:

```py
the_model = torch.load(PATH)
```

However in this case, the serialized data is bound to the specific classes and the exact directory structure used, so it can break in various ways when used in other projects, or after some serious refactors.



### 1.5

antpush TEST folder: set succeeded reward to be the distance between init node and goal node

antpush TEST2 folder: post-processes value function output as v > 0 ? : 0 : v



ant9rooms TEST2: give additional -1 penalty when close to wall

### 1.31

ant9rooms TEST3: TEST2 with 300 rollouts expanding graph, and 200 rollouts for finetuning

### 2.3

ant9rooms TEST4: TEST2 with 400 rollouts on expanding graph, and 200 rollouts for fine tuning. Also with optimized value function policy data

### 2.7

ant9rooms TEST5 (ant9roomscloseddoors): ant9rooms with some doors closed, 4e5 for expanding graph and 2e5 for fine tuning. Also with optimized value function policy data.

ant9rooms TEST5 (ant9roomscloseddoors) with 4e5 on graph, 2e5 on fine tuning, 4e5 on training SL

ant9rooms TEST5 (ant9roomscloseddoors) with 4e5 on graph, 4e5 on fine tuning, 4e5 on training SL







### 2.11

ant16rooms test1: using same setting as ant9rooms test5 (4e5)

ant16rooms test2: test1 with 6e5 to get graph, 4e5 on finetuning, 4e5 on SL training

​	ant16rooms test2.1: test1 with 6e5 to get graph, 4e5 on finetuning, 8e5 on SL training, random start position

​	ant16rooms test2.2: test1 with 6e5 to get graph, 4e5 on finetuning, 8e5 on SL training, fixed start position

​	ant16rooms test2.3: test1 with 6e5 to get graph, 6e5 on finetuning, _e5 on SL training, random start position

ant16rooms test3: close some doors in the ant16rooms env, do 6e5 on getting graph

**the above tests are based on distance reward and 1000/-1000 if succeeded or not, with back propagation**

- ant16rooms test4: ant16rooms having close some doors with new reward function[binary reward: -1 or 0] and 8e6 on getting graph, 6e5 on finetuning, 6e5 on SL(**without** resetting desired goal)
  - with value function policy replay buffer size 2e5 timesteps and update frequency every 10 steps
- ant16rooms test5: ant16roomscloseddoors env with well expanded graph. 8e5 on finetuning, 8e5 on SL
- ant16rooms test6: ant16roomscloseddoors with binary reward + -1000 penalty for point in a narrow region(norm2=2) in failed traces, 8e5 generating graph
- ant16rooms test7: ant16roomscloseddoors with binary reward + -1000 penalty for point in a narrow region(norm2=2) in failed traces, **also with** selecting closest node by distance

ant16rooms test8 1): **test** latest algo on ant16rooms [with 0,0,...,0/1 rewardm, **also with** selecting closest node by distance [only at step2 and 3] + #succeed traces * 0.8 as train data in SL 

ant16rooms test8 2): **test** latest algo on ant16rooms [with 0,0,...,0/1 reward, **also with** selecting closest node by distance [only at step2 and 3] + -1 at last states in a small region + #succeed traces * 0.8 as train data in SL 

ant16rooms test8 3): **test** latest algo on ant16rooms [with 0,0,...,0/1 reward, **also with** selecting closest node by distance [only at step2 and 3] + -1 at last states in a small region + use all traces as train data in SL 

ant16rooms test8 4): **test** latest algo on ant16rooms [with 0,0,...,0/1 reward, **also with** selecting closest node by distance [only at step2 and 3] + -1 at last states in a small region + use all SUCCEEDED traces as train data in SL 

[IMPORTANT] all previous SL phase based on selecting closest node by value. So, need of rerun

ant16rooms test8 5): **test** latest algo on ant16rooms [with 0,0,...,0/1 reward, **also with** selecting closest node by distance [only at step2 and 3] + -1 at last states in a small region + use all SUCCEEDED traces as train data in SL [IMPORTANT] test 8.5 has by_distance mode to select path for SL



# ant16rooms test9
it is actually test8 graph, but will redo finetuning and SL


0.634, 0.678














1. merge repo
2. together readme
3. result of single path
4. gif of examples


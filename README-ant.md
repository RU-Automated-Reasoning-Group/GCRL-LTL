# GCRL-LTL

## Setup

Environment for Ant16rooms experiment is developed mainly based on the following version of packages:
```
	numpy=1.18.5
	torch=1.5.1
	gym=0.13.1
	mujoco_py=2.0.2.5
```
along with MuJoCo simulator version `mujoco200` from [MuJoCo release website](https://www.roboti.us/download.html)

1. Docker
	Out environment is designed based on the environment used in [GCSL](https://github.com/dibyaghosh/gcsl). To download the docker image:
	
	```
	docker pull dibyaghosh/gcsl:0.1
	```
2. Conda
	For conda environment setting-up, please refer to [conda_environment.yml](.\ant\misc\conda_environment.yml) for all specific version of packages.
3. Python(pip)
	For python pip packages, please refer to [python_requirement.txt](.\ant\misc\python_requirement.txt) for all specific version of packages.
	
	
## run

Add workspace directory to PYTHONPATH:
```
export PYTHONPATH="${PYTHONPATH}:{path_of_GCRL-LTL_ant_folder}"
```



##### step1: Generate graph

##### step2: Finetune value policy

##### step3: reinforcement learning

## To provide new spec 

##### Definition of items

##### Combine items together

##### Examples 



## Specifications and results shown in Gifs

##### specification phi1

##### specification phi2

##### specification phi3

##### specification phi4

##### specification phi5


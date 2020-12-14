# Udacity - Deep Reinforcement Learning Nanodegree (Continuous Control)

### Project Details

This is the second project of the Deep Reinforcement Learning Nanodegree. I trained a DDPG Agent to solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.  This project is influenced by the previous one: https://github.com/escribano89/bananas-dqn and the DDPG implementations from the Udacity's repository https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum.

In this environment, a double-jointed arm can move to target locations. A reward is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, there are two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

I opted for solving the second version of the problem explained below:

#### Option 2: Solve the Second Version

your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
- After each episode, I add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. I then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Requirements
In order to prepare the environment, follow the next steps after downloading this repository:
* Create a new environment:
	* __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	* __Windows__: 
	```bash
	conda create --name dqn python=3.6 
	activate drlnd
	```
* Min install of OpenAI gym
	* If using __Windows__, 
		* download [swig for windows](http://www.swig.org/Doc1.3/Windows.html) and add it the PATH of windows
		* install [ Microsoft Visual C++ Build Tools ](https://visualstudio.microsoft.com/es/downloads/).
	* then run these commands
	```bash
	pip install gym
	pip install gym[classic_control]
	pip install gym[box2d]
	```
* Install the dependencies under the folder [python/] (https://github.com/udacity/deep-reinforcement-learning/tree/master/python).
```bash
	cd python
	pip install .
```
* Create an IPython kernel for the `drlnd` environment
```bash
	python -m ipykernel install --user --name drlnd --display-name "drlnd"
```


* Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)


* Unzip the downloaded file and move it inside the project's root directory
* Change the kernel of you environment to `drlnd`
* Open the **train.py** and the **test.py** files and change the path to the unity environment appropriately (UNITY_EXE_PATH=PATH_OF_THE_REACHER_EXE)

## Getting started

If you want to test the trained agent, execute the **test.py** file. 

If you want to train the agent, execute the **train.py** file. After reaching the goal, the networks weights will be stored in the project's root folder.


## Resources

* report.pdf: A document that describes the details of the implementation and future proposals.
* agent: implemented agent using the DDPG algorithm (without exploration noise)
* actor: the actor NN model
* critic: the critic NN model
* unity_env: a class for handling the unity environment
* replay_buffer: a class for handling the experience replay
* test.py: Entry point for testing the agent using the trained networks
* actor_theta.pth, critic_theta.pth: Our model's weights ***(Solved in less than 120 episodes)***

## Trace of the training

![Training](https://github.com/escribano89/reacher-ddpg/blob/main/score.PNG)

![Training](https://github.com/escribano89/reacher-ddpg/blob/main/trace.PNG)

## Video

You can find an example of the trained agent [here](https://youtu.be/Lm9tgbPyDFM)

[![Navigation](https://img.youtube.com/vi/Lm9tgbPyDFM/0.jpg)](https://youtu.be/Lm9tgbPyDFM)

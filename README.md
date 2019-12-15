
# Project 3: Collaboration and Competition

### Introduction

For this project, the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. There are 2 agents (racquets) in this environment and the goal is to make them pass the ball among them over the net as long as possible.

The task is episodic and ends if the ball is hit out of bounds or the ball hits the ground or the time runs out. 

#### Reward
- Each agent receives a reward of 0.1 if it hits the ball correctly across the net and withing the playing area.
- Each agent receives a reward  of -0.01 if it drops the ball to the ground or hits it out of bounds.
#### States
- The agents and the ball's states are described by a vector of 8 elements corresponding to to position and velocities.
- Each agent receives an observation of 24 elements (state of itself, state of the ball and state of the other agent)

#### Actions
- Each action is a vector with 2 numbers, corresponding to movement toward net or away from net, and jumping.

### Success criteria for Training
- After after episode (game), the cumulative reward for each agent is determined. The maximum of the cumulative rewards of the agents for an episode averaged over the most recent 100 episodes needs to be higher than 0.5 for the training to be successful.
 

### Setting Up the Python Environment
Follow the instructions below to set up your python environment to run the code in this repository, 
1. Create (and activate) a new environment with Python 3.6.
	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Clone the repository [DRLND](https://github.com/udacity/deep-reinforcement-learning), and navigate to the `python/` folder.  Then, install several dependencies.
      -  ```bash
         git clone https://github.com/udacity/deep-reinforcement-learning.git
         cd deep-reinforcement-learning/python
         pip install .
         ```
4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.
     -  ```bash
          python -m ipykernel install --user --name drlnd --display-name "drlnd"
        ```
5. Before running code in the notebooks, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

### Setting Up the Unity Environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
     - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
     - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
     - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
     - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the directory where you have cloned this project, and unzip (or decompress) the file.
3. Name the directory where the Unity environment is unzipped as `Tennis.app`
4. The repo contains the zip files for the Mac OSX operating system.

### Instructions
Run the `Collab_Compet_Submission.ipynb` notebook to get started.  
1. Navigate to the root directory of this repo. 
2. Start the Jupyter Notebook in the activated `drlnd` conda environment
   - ```bash
     jupyter notebook
     ```
3. Open  `Collab_Compet_Submission.ipynb` in the Jupyter file browser  
  

### Implementation Details
The details can be found in `REPORT.ipynb` file






# DRLND-project2-continous-control

In this project agents are trained to bounce a ball over the net.

## Enviroment

State space in this enviroment has 24 dimentions corresponding to position, velocity of ball and racket. Given this information each agent has to keep the ball in play.

Action space is continous and consists of a 2 number vector corresponding to movement to and away from the net and jumping. 

The task is solved when the average score of the agent in 100 consecutive episodes is 0.5 or higher.

## Setup

To run the code, either to train or check trained agent, clone the repository and install requirements using:

```bash
pip install -r requirements.txt
```

This task was solved using Python 3.9 in order to use newer version of pytorch that would cooperate with CUDA 116

Then download the unity enviorment from the link below and unzip it in into the cloned repository. Links provided below are for the 20 agent version of the enviroment.

Download the environment from one of the links below.  You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Run

To train agent run cells from Tennis.ipynb notebook

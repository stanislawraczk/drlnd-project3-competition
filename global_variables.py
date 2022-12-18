from math import exp, log

num_steps = 5000 # number of steps with eps not min, approx 1500 steps per 100 episode in early stages
SEED = 0
LR_ACTOR = 1e-04
LR_CRITIC = 1e-04
TAU = 1e-01
GAMMA = 0.99
BATCH_SIZE = 512
LEARN_EVERY = 20
LEARN_NUM = 10
BUFFER_SIZE = int(1e6)
EPSILON = 1
EPS_MIN = 1e-02
EPS_DECAY = exp(log(EPS_MIN/EPSILON)/num_steps)
MU = 0.
THETA = .2
SIGMA = .15
WEIGHT_DECAY = 0
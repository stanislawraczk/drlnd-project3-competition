from math import exp, log

num_steps = 5000 # number of steps with eps not min, approx 1500 steps per 100 episode in early stages
SEED = 0
LR_ACTOR = 1e-04 # actor learning rate
LR_CRITIC = 1e-04 # critic learning rate
TAU = 1e-01 # soft update parameter
GAMMA = 0.99 # discount parameter
BATCH_SIZE = 512
LEARN_EVERY = 20 # number of steps between learning
LEARN_NUM = 10 # number of times the networks are optimized each learning step
BUFFER_SIZE = int(1e6) # memory size
EPSILON = 1 # starting epsilon value
EPS_MIN = 1e-02 # minimal epsilon value
EPS_DECAY = exp(log(EPS_MIN/EPSILON)/num_steps) # epsilon should hit eps_min after 5000 steps (num_steps variable)
MU = 0. # OU Noise mu parameter
THETA = .2 # OU Noise theta parameter
SIGMA = .15 # OU Noise sigma parameter
WEIGHT_DECAY = 0 # weight decay turned off
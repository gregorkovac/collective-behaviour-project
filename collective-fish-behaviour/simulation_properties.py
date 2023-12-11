# Screen properties
import numpy as np

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
FRAME_RATE = 60

# Scene manager properties
NUM_FISH = 50

# Fish properties
INITIAL_SPEED = 3  # in m/s

# Parameters for hydrodynamic interactions
K_P = 0.8 # Attraction factor
K_V = 0.1 # Alignment factor
SIGMA = 0.1 # Gaussian-distributed rotational noise
K_NN = 10 # k nearest neighbours

R_0 = 0.8  # Length of a fish in mters
S = np.pi * (R_0 / 2) ** 2  # Surface area assuming fish is a circle with radius r_0/2
I_F = S * K_P / INITIAL_SPEED # Dipole intestity
I_PARALLEL = K_V * np.sqrt(INITIAL_SPEED / K_P)
I_N = SIGMA * np.power(INITIAL_SPEED * K_P, -1/4)
#I_F = 0.01
# print(I_F)
AQUARIUM_WIDTH = 30 # Width of the aquarium in meters
AQUARIUM_HEIGHT = 30  # Height of the aquarium in meters
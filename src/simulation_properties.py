# Screen properties
import numpy as np

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
FRAME_RATE = 60

# Scene manager properties
NUM_FISH = 20

# Fish properties
INITIAL_SPEED = 6.5  # in m/s

# Parameters for hydrodynamic interactions
K_P = 0.0 # Attraction factor
K_V = 0.1 # Alignment factor
K_S = 0.4 # Separation factor
K_C = 0.4 # Cohesion factor
SIGMA = 0.5 # Gaussian-distributed rotational noise
K_NN = 2 # k nearest neighbours

###### PHASES ######
# SWARMING 
# All parameters are low (0) except for separation (for example 0.1)
#
# SCHOOLING
# Alignment is >= 0.1, separation is low (<= 0.1), cohesion is low (<= 0.1), separation os low (<= 0.1)
#
# MILLING
# ?
#
# TURNING
# Same as schooling but with high attraction, for example:
# K_P = 10
# K_V = 0.4
# K_S = 0.1
# K_C = 0.1



R_0 = 0.8  # Length of a fish in mters
S = np.pi * (R_0 / 2) ** 2  # Surface area assuming fish is a circle with radius r_0/2
I_F = S * K_P / INITIAL_SPEED # Dipole intestity
# I_PARALLEL = K_V * np.sqrt(INITIAL_SPEED / K_P)
# I_N = SIGMA * np.power(INITIAL_SPEED * K_P, -1/4)

I_N = 0.5

print(I_N)

I_PARALLEL = 0
# I_N = 0
#I_F = 0.01
# print(I_F)
AQUARIUM_WIDTH = 30 # Width of the aquarium in meters
AQUARIUM_HEIGHT = 30  # Height of the aquarium in meters
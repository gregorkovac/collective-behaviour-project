import numpy as np

from simulation_properties import *


class Fish:
    def __init__(self, initial_position):
        self.location = initial_position
        self.velocity = INITIAL_SPEED
        # Generate random unit vector representing direction
        self.direction = np.linalg.norm(np.random.rand(2))

    def tick(self, delta_time, neighbor_fishes):
        for other_fish in neighbor_fishes:
            direction = other_fish.get_location() - self.location
            if np.linalg.norm(direction) != 0:
                norm_direction = direction / np.linalg.norm(direction)
            else:
                norm_direction = 1

            # Update fish location
            self.location += delta_time * self.velocity * norm_direction

    def get_location(self):
        return self.location

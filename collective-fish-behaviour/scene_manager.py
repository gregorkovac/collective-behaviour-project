import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

from simulation_properties import *
from fish import Fish


# Scene manager operates with normalised screen width
class SceneManager:
    def __init__(self, aspect_ratio):
        self.num_fish = NUM_FISH
        self.fishes = []
        self.aspect_ratio = aspect_ratio

        # TODO: Implement container size in meters?
        # Generate starting fish positions
        self.fish_positions = self.generate_random_vectors_2D(self.num_fish, 1.0, self.aspect_ratio)

    def initialize(self):
        # Create fishes
        for i in range(self.num_fish):
            self.fishes.append(Fish(self.fish_positions[i]))

    def tick(self, delta_time):
        # Create voronoi diagram
        vor = Voronoi(self.fish_positions)

        # Plot the Voronoi diagram - test only
        # voronoi_plot_2d(vor)
        # plt.show()

        # Update each fish
        for index, fish in enumerate(self.fishes):
            # TODO: Get neighbors from vor diagram. Currently is only the closest neighbour.

            # Find neighbour fishes
            neighbour_fishes = []
            index = self.find_nearest_fish_neighbour(fish.get_location())
            neighbour_fishes.append(self.fishes[index])

            # Call fish tick
            fish.tick(delta_time, neighbour_fishes)

            # Update global fish location with new location
            self.fish_positions[index] = fish.get_location()

    def find_nearest_fish_neighbour(self, fish_location):
        distances = np.linalg.norm(self.fish_positions - fish_location, axis=1)
        # Find the index of the first closest neighbor
        closest_index = np.argmin(distances)

        # Set the distance of the first closest neighbor to infinity to exclude it
        distances[closest_index] = np.inf

        # Return the index of the second-closest neighbor
        return np.argmin(distances)

    def get_fish_locations(self):
        return self.fish_positions

    @staticmethod
    def generate_random_vectors_2D(num_points, max_x, max_y):
        return np.random.uniform(0, 1, size=(num_points, 2)) * np.array([max_x, max_y])

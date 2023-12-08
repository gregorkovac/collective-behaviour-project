from scipy.spatial import Voronoi
from simulation_properties import *


# Scene manager operates with normalised screen width
class SceneManager:
    def __init__(self):
        self.num_fish = NUM_FISH
        self.aquarium_width = AQUARIUM_WIDTH
        self.aquarium_height = AQUARIUM_HEIGHT
        # Array structure: [x_position, y_position, velocity, direction_x, direction_y]
        self.fishes = np.zeros((self.num_fish, 5), dtype=float)

        # Parameters for hydrodynamic interactions
        self.k_p = K_P  # Attraction factor
        self.k_v = K_V  # Alignment factor
        self.sigma = SIGMA  # Gaussian-distributed rotational noise
        self.r_0 = R_0  # Length of a fish (in normalized units)
        self.S = S  # Surface area assuming fish is a circle with radius r_0/2
        self.I_f=I_F # Dipole intensity

    def initialize(self):
        # Create fishes
        center_x = self.aquarium_width / 2
        center_y = self.aquarium_height / 2

        # Generate initial positions within 1/5th of the aquarium's size, centered
        initial_positions = self.generate_random_vectors_2D(self.num_fish,
                                                            center_x - self.aquarium_width / 10,
                                                            center_x + self.aquarium_width / 10,
                                                            center_y - self.aquarium_height / 10,
                                                            center_y + self.aquarium_height / 10)

        self.fishes[:, :2] = initial_positions
        self.fishes[:, 2] = INITIAL_SPEED


        # Initialize directions (random unit vectors)
        angles = np.random.rand(self.num_fish) * 2 * np.pi
       # print(angles)
        self.fishes[:, 3] = np.cos(angles)
        self.fishes[:, 4] = np.sin(angles)

        # Initialize directions (random unit vectors)
        #random_angle = np.random.rand() * 2 * np.pi  # Generate one random angle
        #print(random_angle)
        #self.fishes[:, 3] = np.cos(random_angle)  # Assign cosine of the angle to all fishes
        #self.fishes[:, 4] = np.sin(random_angle)

    def tick(self, delta_time):
        # self.fishes[:, 0] += self.fishes[:, 2] * self.fishes[:, 3] * delta_time
        # self.fishes[:, 1] += self.fishes[:, 2] * self.fishes[:, 4] * delta_time

        # Handle edge collisions
        for i in range(self.num_fish):
            # Check for collision with horizontal boundaries
            if self.fishes[i, 0] <= 0 or self.fishes[i, 0] >= self.aquarium_width:
                self.fishes[i, 3] *= -1  # Reverse horizontal direction

            # Check for collision with vertical boundaries
            if self.fishes[i, 1] <= 0 or self.fishes[i, 1] >= self.aquarium_height:
                self.fishes[i, 4] *= -1  # Reverse vertical direction

            # Ensure the position stays within bounds
            self.fishes[i, 0] = np.clip(self.fishes[i, 0], 0, self.aquarium_width)
            self.fishes[i, 1] = np.clip(self.fishes[i, 1], 0, self.aquarium_height)

        # Compute Voronoi diagram
        vor = Voronoi(self.fishes[:, :2])

        for i in range(self.num_fish):
            U_i = np.zeros(2, dtype=float)  # Initialize as a float array
            Omega_i = 0.0  # Initialize rotational influence

            neighbors = self.find_voronoi_neighbors(i, vor)

            e_i_parallel = self.fishes[i, 3:5]  # Current direction of fish i
            e_i_perpendicular = np.array([-e_i_parallel[1], e_i_parallel[0]])  # Perpendicular to e_i_parallel

            # Check for collisions and update positions
            self.handle_collisions(i)

            for j in neighbors:
                if i != j:
                    # Relative position vector from fish i to fish j
                    relative_pos = self.fishes[j, :2] - self.fishes[i, :2]
                    rho_ij = np.linalg.norm(relative_pos)

                    if rho_ij != 0:
                        # Convert to polar coordinates
                        theta_ji = np.arctan2(relative_pos[1], relative_pos[0])
                        rho_ji = rho_ij


                        # Compute u_ji
                        e_j_theta = np.array([np.cos(theta_ji), np.sin(theta_ji)])  # Direction vector of fish j
                        u_ji = self.I_f / np.pi * (
                                e_j_theta * np.sin(theta_ji) + np.array([-np.sin(rho_ji), np.cos(rho_ji)]) * np.sin(
                            rho_ji)) / rho_ij ** 2

                        U_i += u_ji

                        u_ji_gradient = self.calculate_gradient_u_ji(i, j)
                        Omega_i += np.dot(e_i_parallel, u_ji_gradient) * np.dot(e_i_perpendicular, u_ji_gradient)


            # Compute the orientation update
            theta_i_update = self.calculate_orientation_update(i, neighbors, Omega_i)
            # Update the orientation of fish i
            current_direction = self.fishes[i, 3:5]
            new_orientation = np.arctan2(current_direction[1], current_direction[0]) + theta_i_update

            # Update direction vector based on new orientation
            self.fishes[i, 3] = np.cos(new_orientation)
            self.fishes[i, 4] = np.sin(new_orientation)

            # Update position
            self.fishes[i, 0] += (self.fishes[i, 2] * self.fishes[i, 3] + U_i[0]) * delta_time
            self.fishes[i, 1] += (self.fishes[i, 2] * self.fishes[i, 4] + U_i[1]) * delta_time

    def find_voronoi_neighbors(self, fish_index, vor):
        neighbors = set()
        point_region = vor.point_region[fish_index]
        vertices = vor.regions[point_region]

        for vertex in vertices:
            if vertex >= 0:  # Ignore vertices at infinity
                # Find which regions/points are adjacent to this vertex
                for region_index in range(len(vor.point_region)):
                    if vertex in vor.regions[vor.point_region[region_index]]:
                        neighbor_index = region_index
                        if neighbor_index != fish_index and neighbor_index not in neighbors:
                            neighbors.add(neighbor_index)

        return list(neighbors)

    def calculate_orientation_update(self, fish_index, neighbors, Omega_i):
        # Initialize variables for averaging
        avg_influence = 0.0
        total_weight = 0.0

        for neighbor_idx in neighbors:
            # Calculate the relative orientation and position
            relative_pos = self.fishes[neighbor_idx, :2] - self.fishes[fish_index, :2]
            distance = np.linalg.norm(relative_pos)

            if distance != 0:
                relative_orientation = np.arctan2(relative_pos[1], relative_pos[0])
                influence = self.calculate_influence(fish_index, neighbor_idx, distance, relative_orientation)

                # Weight influence by distance or another metric
                weight = 1 + np.cos(relative_orientation)
                avg_influence += influence * weight
                total_weight += weight

        if total_weight > 0:
            avg_influence /= total_weight

        # Add Gaussian-distributed noise
        noise = np.random.normal(0, self.sigma)

        # Calculate total orientation update
        theta_i_update = avg_influence + noise + Omega_i

        return theta_i_update

    def calculate_influence(self, fish_index, neighbor_idx, distance, relative_orientation):
        # Extract the direction vectors of both fish
        dir_fish = self.fishes[fish_index, 3:5]
        dir_neighbor = self.fishes[neighbor_idx, 3:5]

        # Calculate alignment influence
        alignment_influence = np.dot(dir_fish, dir_neighbor)

        # Adjust alignment influence based on relative orientation
        # For example, influence is stronger when fish are facing each other
        alignment_adjustment = np.cos(relative_orientation)
        adjusted_alignment_influence = self.k_v * alignment_influence * alignment_adjustment

        return adjusted_alignment_influence

    def calculate_gradient_u_ji(self, i, j):
        # Relative position vector from fish i to fish j
        relative_pos = self.fishes[j, :2] - self.fishes[i, :2]
        distance = np.linalg.norm(relative_pos)

        if distance == 0:
            return np.zeros(2)  # Avoid division by zero

        # Direction from fish i to fish j, rotated by 90 degrees to represent a vortex
        direction = np.array([-relative_pos[1], relative_pos[0]]) / distance

        # For a vortex, the velocity field (and therefore the gradient) varies inversely with distance
        gradient = direction / distance  # Inverse distance relationship

        return gradient

    def handle_collisions(self, fish_index):
        for other_index in range(self.num_fish):
            if fish_index != other_index:
                distance = np.linalg.norm(self.fishes[fish_index, :2] - self.fishes[other_index, :2])
                min_distance = self.r_0  # minimum distance to avoid collision, can be radius of fish

                # If fishes are too close, push them away from each other
                if distance < min_distance:
                    overlap = min_distance - distance
                    direction = (self.fishes[fish_index, :2] - self.fishes[other_index, :2]) / distance
                    self.fishes[fish_index, :2] += direction * overlap * 0.5  # Adjust position by half the overlap
                    self.fishes[other_index, :2] -= direction * overlap * 0.5

    def get_fish_locations(self):
        return self.fishes[:, :2]

    @staticmethod
    def generate_random_vectors_2D(num_points, min_x, max_x, min_y, max_y):
        x_positions = np.random.uniform(min_x, max_x, num_points)
        y_positions = np.random.uniform(min_y, max_y, num_points)
        return np.stack((x_positions, y_positions), axis=-1)


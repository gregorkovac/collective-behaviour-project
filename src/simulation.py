import numpy as np
from simulation_parameters import SP

class Response:
    def __init__(self, pos, dir):
        self.pos = pos
        self.dir = dir

class Simulation:
    def __init__(self):

        self.ang2dir = lambda x: np.array([np.cos(x), np.sin(x)]).T
        self.dir2ang = lambda x: np.arctan2(x[:, 1], x[:, 0])

        spawn_offset = 5
        self.pos = np.random.uniform(spawn_offset, np.array(SP.aquarium_size)-spawn_offset, (SP.num_fish, 2))
        self.dir = self.ang2dir(np.random.rand(SP.num_fish) * 2 * np.pi)
        
        self.dists = self.calculate_distances()
    
    def simulate(self, deltaTime, params):
        # consider separation factor for direction 
        separation = self.separation(params.k_s, params.separation_distance)

        # consider alignment factor for direction
        alignment = self.alignment(params.k_v, params.alignment_distance)

        # consider cohesion factor for direction
        cohesion = self.cohesion(params.k_c, params.cohesion_distance)

        # sum all factors
        if len(separation) > 0:
            self.dir[:len(separation)] += separation * deltaTime
        if len(alignment) > 0:
            self.dir[:len(alignment)] += alignment * deltaTime
        if len(cohesion) > 0:
            self.dir[:len(cohesion)] += cohesion * deltaTime
        
        # normalize new direction
        self.dir /= np.linalg.norm(self.dir, axis=1)[:, np.newaxis]

        # Update position based on new direction and constant speed
        self.pos = self.pos + self.dir * params.vel * deltaTime

        match params.borders:
            # Wrap around the aquarium
            case "loop":
                self.pos = np.where(self.pos < 0, self.pos + SP.aquarium_size, self.pos)
                self.pos = np.where(self.pos > SP.aquarium_size, self.pos - SP.aquarium_size, self.pos)
            # Bounce off the walls
            case "bounce":
                self.dir = np.where(self.pos < 0, np.abs(self.dir), self.dir)
                self.dir = np.where(self.pos > SP.aquarium_size, -np.abs(self.dir), self.dir)
                self.pos = np.where(self.pos < 0, 0, self.pos)
                self.pos = np.where(self.pos > SP.aquarium_size, SP.aquarium_size, self.pos)
            # Repel from the walls
            case "repulsion":
                mask1 = self.pos < 1/10*SP.aquarium_size[0]
                mask2 = self.pos > 9/10*SP.aquarium_size[0]
                self.dir[mask1] += (1/10*SP.aquarium_size[0] - self.pos[mask1]) * params.vel * deltaTime
                self.dir[mask2] -= (self.pos[mask2] - 9/10*SP.aquarium_size[0]) * params.vel * deltaTime

        # Update distances between all fish
        self.dists = self.calculate_distances()

        # Handle collisions
        if params.collisions:
            self.resolve_collisions()

        return Response(self.pos, self.dir)
    
    def flow(self, k_p, f_d):
        idx = np.argwhere(self.dists < f_d)
        idx = idx[idx[:, 0] != idx[:, 1]]
        if len(idx) > 0:
            pass
    
    def separation(self, k_s, s_d):
        idx = np.argwhere(self.dists < s_d)
        idx = idx[idx[:, 0] != idx[:, 1]]
        if len(idx) > 0:
            direction = self.pos[idx[:, 0]] - self.pos[idx[:, 1]]
            norm = np.linalg.norm(direction, axis=1)
            norm = np.where(norm > 0, norm, 1)
            direction /= norm[:, np.newaxis]
            dir_x = np.bincount(idx[:, 0], weights=direction[:, 0])
            dir_y = np.bincount(idx[:, 0], weights=direction[:, 1])
            dir = np.column_stack((dir_x, dir_y))
            return dir * k_s
        return []
    
    def alignment(self, k_v, a_d):
        idx = np.argwhere(self.dists < a_d)
        idx = idx[idx[:, 0] != idx[:, 1]]
        if len(idx) > 0:
            dir_x = np.bincount(idx[:, 0], weights=self.dir[idx[:, 1], 0])
            dir_y = np.bincount(idx[:, 0], weights=self.dir[idx[:, 1], 1])
            count = np.bincount(idx[:, 0])
            count = np.where(count > 0, count, 1)
            dir = np.column_stack((dir_x / count, dir_y / count))
            return (dir - self.dir[:len(dir)]) * k_v
        return []
    
    def cohesion(self, k_c, c_d):
        idx = np.argwhere(self.dists < c_d)
        idx = idx[idx[:, 0] != idx[:, 1]]
        if len(idx) > 0:
            pos_x = np.bincount(idx[:, 0], weights=self.pos[idx[:, 1], 0])
            pos_y = np.bincount(idx[:, 0], weights=self.pos[idx[:, 1], 1])
            count = np.bincount(idx[:, 0])
            count = np.where(count > 0, count, 1)
            pos = np.column_stack((pos_x / count, pos_y / count))
            return (pos - self.pos[:len(pos)]) * k_c
        return []
    
    def calculate_distances(self):
        pos_rep = self.pos[:, :, np.newaxis].repeat(SP.num_fish, axis=2).transpose((2, 1, 0))
        dists = np.linalg.norm(pos_rep - self.pos[:,:, np.newaxis], axis=1)
        return dists
    
    def resolve_collisions(self):
        min_dist = 2*SP.fish_radius
        idx = np.argwhere(self.dists < min_dist)
        idx = idx[idx[:, 0] != idx[:, 1]]
        if len(idx) > 0:
            overlap = min_dist - self.dists[idx[:, 0], idx[:, 1]]
            direction = self.pos[idx[:, 0]] - self.pos[idx[:, 1]]
            norm = np.linalg.norm(direction, axis=1)
            norm = np.where(norm > 0, norm, 1)
            direction /= norm[:, np.newaxis]
            self.pos[idx[:, 0]] += direction * overlap[:, np.newaxis] * 0.5
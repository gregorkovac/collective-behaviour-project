import numpy as np
from simulation_parameters import SP
from scipy.spatial import Voronoi

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
        self.neighbours = None
    
    def simulate(self, deltaTime, params):
        # get neighbours
        self.neighbours = self.get_voronoi_neighbours()

        # consider separation factor for direction 
        separation = self.separation(params.k_s, params.separation_distance)

        # consider alignment factor for direction
        alignment = self.alignment(params.k_a, params.alignment_distance)

        # consider cohesion factor for direction
        cohesion = self.cohesion(params.k_c, params.cohesion_distance)
        
        # consider flow factor for direction
        flow = self.flow(params.vel, params.k_v, params.k_p, params.flow_distance)

        # consider flow offset factor for direction
        flow_offset = self.flow_offset(params)


        # consider alignment attraction factor for direction
        alignment_attraction = self.alignment_attraction(params)

        # consider standard wiener process term with standard deviation params.sigma
        wiener = self.wiener(params)

        # sum all factors
        #if len(separation) > 0:
        #    self.dir[:len(separation)] += separation * deltaTime
        #if len(alignment) > 0:
        #    self.dir[:len(alignment)] += alignment * deltaTime
        #if len(cohesion) > 0:
        #    self.dir[:len(cohesion)] += cohesion * deltaTime
        #if len(flow) > 0:
        #    alpha = self.dir2ang(self.dir[flow[1]])
        #    alpha += -params.vel * flow[0] * deltaTime
        #    self.dir[flow[1]] = self.ang2dir(alpha)

        # update direction
        alpha = self.dir2ang(self.dir)
        alpha += (alignment_attraction + wiener) * deltaTime
        self.dir[:len(alignment_attraction)] = self.ang2dir(alpha)
        

        
        
        
        

        # normalize new direction
        self.dir /= np.linalg.norm(self.dir, axis=1)[:, np.newaxis]

        # Update position based on new direction and constant speed
        self.pos = self.pos + (self.dir + flow_offset) * params.vel * deltaTime

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
    
    def alignment_attraction(self, params):
        idx = self.neighbours
        if len(idx) > 0:
            e_ji = self.pos[idx[:, 0]] - self.pos[idx[:, 1]]
            dist = np.linalg.norm(e_ji, axis=1)
            dist = np.where(dist > 0, dist, 1)
            e_ji /= dist[:, np.newaxis]

            e_i = self.dir[idx[:, 0]]
            e_j = self.dir[idx[:, 1]]
            theta_ij = np.arccos((-e_ji * e_i).sum(axis=1))
            theta_ji = np.arccos((e_ji * e_j).sum(axis=1))
            
            phi_ij = np.pi - theta_ij - theta_ji

            I_paralell = params.k_v * np.sqrt(params.vel/params.k_p)

            aa = dist * np.sin(theta_ij) + I_paralell * np.sin(phi_ij)

            weights = 1 + np.cos(theta_ij) + 1e-6
            aa_w = np.bincount(idx[:, 0], weights=aa*weights, minlength=SP.num_fish)
            aa_w /= np.bincount(idx[:, 0], weights=weights, minlength=SP.num_fish)
            return aa_w
        
    def wiener(self, params):
        I_n = SP.sigma * np.power(params.vel*params.k_p, -1/4)
        return np.random.normal(0, 1, (SP.num_fish, 2)) * I_n
    
    def flow_offset(self, params):
        idx = np.argwhere(self.dists < params.flow_distance)
        idx = idx[idx[:, 0] != idx[:, 1]]
        if len(idx) > 0:
            e_ji = self.pos[idx[:, 0]] - self.pos[idx[:, 1]]
            dist = np.linalg.norm(e_ji, axis=1)
            dist = np.where(dist > 0, dist, 1)
            e_ji /= dist[:, np.newaxis]

            e_j = self.dir[idx[:, 1]]

            theta_ji = np.arccos((e_ji * e_j).sum(axis=1))

            u = e_j * np.sin(theta_ji)[:, np.newaxis] + e_ji * np.cos(theta_ji)[:, np.newaxis]
            u = u / (dist**2)[:, np.newaxis]
            u_x = np.bincount(idx[:, 0], weights=u[:, 0])
            u_y = np.bincount(idx[:, 0], weights=u[:, 1])
            U = np.column_stack((u_x, u_y))
            if len(U) < SP.num_fish:
                print(U)
                print(np.zeros((SP.num_fish - len(U), 2)))
                U = np.concatenate((U, np.zeros((SP.num_fish - len(U), 2))))
            factor = SP.fish_radius**2 * params.k_p / params.vel
            return U * factor
    
    def flow(self, v, k_v, k_p, f_d):
        idx = np.argwhere(self.dists < f_d)
        idx = idx[idx[:, 0] != idx[:, 1]]
        if len(idx) > 0:
            e_ji = self.pos[idx[:, 0]] - self.pos[idx[:, 1]]
            dist = np.linalg.norm(e_ji, axis=1)
            dist = np.where(dist > 0, dist, 1)
            e_ji /= dist[:, np.newaxis]

            e_i = self.dir[idx[:, 0]]
            e_j = self.dir[idx[:, 1]]

            theta_ij = np.arccos((-e_ji * e_i).sum(axis=1))
            theta_ji = np.arccos((e_ji * e_j).sum(axis=1))

            phi_ij = np.pi - theta_ij - theta_ji
            #phi_ij = np.arccos((e_i * e_j).sum(axis=1))

            w = k_v * v * np.sin(phi_ij) + k_p * dist * np.sin(theta_ij)

            w = np.bincount(idx[:, 0], weights=w)
            count = np.bincount(idx[:, 0])
            count = np.where(count > 0, count, 1)
            w /= count
            idx2 = np.where(w > 0)
            return (self.dir2ang(self.dir[idx2]) - w[idx2], idx2)
        return []
    
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
    
    def alignment(self, k_a, a_d):
        idx = np.argwhere(self.dists < a_d)
        idx = idx[idx[:, 0] != idx[:, 1]]
        if len(idx) > 0:
            dir_x = np.bincount(idx[:, 0], weights=self.dir[idx[:, 1], 0])
            dir_y = np.bincount(idx[:, 0], weights=self.dir[idx[:, 1], 1])
            count = np.bincount(idx[:, 0])
            count = np.where(count > 0, count, 1)
            dir = np.column_stack((dir_x / count, dir_y / count))
            return (dir - self.dir[:len(dir)]) * k_a
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
    
    def get_voronoi_neighbours(self):
        vor = Voronoi(self.pos)
        point_region = vor.point_region
        regions = vor.regions
        point_region_length = len(point_region)

        neighbors = []
        for fish_index in range(SP.num_fish):
            for vertex in vor.regions[point_region[fish_index]]:
                if vertex >= 0:
                    for region_index in range(point_region_length):
                        if vertex in regions[point_region[region_index]]:
                            neighbor_index = region_index
                            if neighbor_index != fish_index:
                                neighbors.append((fish_index, neighbor_index))
        return np.array(neighbors)
    
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
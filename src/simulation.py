import numpy as np
from simulation_parameters import SP
from scipy.spatial import Delaunay

np.random.seed(0)

class Response:
    def __init__(self, pos, dir):
        self.pos = pos
        self.dir = dir

class SPPProperties:
    def __init__(self, pairs, e_ji, e_ji_orth, dist, e_i, e_j, theta_ij, theta_ji, phi_ij, e_i_orth):
        self.pairs = pairs
        self.e_ji = e_ji
        self.e_ji_orth = e_ji_orth
        self.dist = dist
        self.e_i = e_i
        self.e_j = e_j
        self.theta_ij = theta_ij
        self.theta_ji = theta_ji
        self.phi_ij = phi_ij
        self.e_i_orth = e_i_orth

class Simulation:
    def __init__(self):

        self.ang2dir = lambda x: np.column_stack((np.cos(x), np.sin(x)))
        self.dir2ang = lambda x: np.arctan2(x[:, 1], x[:, 0])

        spawn_offset = 5
        self.pos = np.random.uniform(spawn_offset, np.array(SP.aquarium_size)-spawn_offset, (SP.num_fish, 2))
        self.dir = self.ang2dir(np.random.rand(SP.num_fish) * 2 * np.pi)
        
        self.dists = self.calculate_distances()

        self.phase = ""

    def simulate(self, deltaTime, params):
        # update spp properties	
        sp = self.get_spp_properties(self.pos, self.dir)
        
        # consider alignment attraction factor for direction
        alignment_attraction = self.alignment_attraction(sp, params)

        # consider standard wiener process term with standard deviation params.sigma
        wiener = self.wiener(params)

        # consider flow factor for position offset
        flow_offset, omega = self.flow_offset(sp, params)

        #flow_offset = 0
        # omega = 0

        # print(omega.mean)

        # update direction
        alpha = self.dir2ang(self.dir)
        alpha += (alignment_attraction + wiener + omega) * deltaTime

        self.dir = self.ang2dir(alpha)

        # update position
        self.pos += (self.dir * params.vel + flow_offset) * deltaTime        

        # handle borders
        match params.borders:
            # Wrap around the aquarium
            case "loop":
                self.pos = np.where(self.pos < 0, self.pos + SP.aquarium_size, self.pos)
                self.pos = np.where(self.pos > (SP.aquarium_size), self.pos - SP.aquarium_size, self.pos)
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
                self.dir /= np.linalg.norm(self.dir, axis=1)[:, np.newaxis]

        # update distances between all fish
        self.dists = self.calculate_distances()

        # handle collisions
        if params.collisions:
            for i in range(1):
                self.resolve_collisions(params)
                self.dists = self.calculate_distances()

        self.get_global_order_params(sp, params)

        return Response(self.pos, self.dir)
    
    @staticmethod
    def get_spp_properties(pos, dir):
        # make pairs of all fish
        pairs = np.array(np.meshgrid(np.arange(SP.num_fish), np.arange(SP.num_fish))).T.reshape(-1, 2)

        e_ji = pos[pairs[:, 0]] - pos[pairs[:, 1]]
        dist = np.linalg.norm(e_ji, axis=1)
        dist = np.where(dist > 0, dist, 1)
        e_ji /= dist[:, np.newaxis]
        e_ji_orth = np.column_stack((-e_ji[:, 1], e_ji[:, 0]))

        e_i = dir[pairs[:, 0]]
        e_j = dir[pairs[:, 1]]

        theta_ij = np.arccos((-e_ji * e_i).sum(axis=1))
        theta_ij_sign = e_i * (-e_ji[:, [1, 0]])
        theta_ij_sign = np.sign(theta_ij_sign[:, 0] - theta_ij_sign[:, 1])
        theta_ij *= theta_ij_sign

        theta_ji = np.arccos((e_ji * e_j).sum(axis=1))
        theta_ji_sign = e_j * (e_ji[:, [1, 0]])
        theta_ji_sign = np.sign(theta_ji_sign[:, 0] - theta_ji_sign[:, 1])
        theta_ji *= theta_ji_sign

        phi_ij = np.arccos((e_i * e_j).sum(axis=1))
        phi_ij_sign = e_i * (e_j[:, [1, 0]])
        phi_ij_sign = np.sign(phi_ij_sign[:, 0] - phi_ij_sign[:, 1])
        phi_ij *= phi_ij_sign

        e_i_orth = np.column_stack((-e_i[:, 1], e_i[:, 0]))

        return SPPProperties(pairs, e_ji, e_ji_orth, dist, e_i, e_j, theta_ij, theta_ji, phi_ij, e_i_orth)
    
    def alignment_attraction(self, sp, params):
        ne, ne_idx = self.get_voronoi_neighbours()

        dist = sp.dist[ne_idx]
        theta_ij = sp.theta_ij[ne_idx]
        phi_ij = sp.phi_ij[ne_idx]

        I_paralell = params.k_v * np.sqrt(params.vel/params.k_p)
        # I_paralell = 9

        aa = dist * np.sin(theta_ij) + I_paralell * np.sin(phi_ij)

        weights = 1 + np.cos(theta_ij) + 1e-6
        aa_w = np.bincount(ne[:, 0], weights=aa*weights, minlength=SP.num_fish)
        counts = np.bincount(ne[:, 0], weights=weights, minlength=SP.num_fish)
        aa_w /= np.where(counts != 0, counts, 1)
        return aa_w
        
    def wiener(self, params):
        I_n = params.sigma * np.power(params.vel*params.k_p, -1/4)
        # print("I_n: ", I_n)
        #I_n = 0.5
        return np.random.normal(0, 1, (SP.num_fish)) * I_n
    
    def flow_u(self, sp, params):
        u = sp.e_ji_orth * np.sin(sp.theta_ji)[:, np.newaxis] + sp.e_ji * np.cos(sp.theta_ji)[:, np.newaxis]
        u = u / (sp.dist**2)[:, np.newaxis]
        u[sp.dist <= 2*params.fish_radius] = 0
        return u

    def flow_offset(self, sp, params, delta=1e-6):
        u = self.flow_u(sp, params)
        u_x = np.bincount(sp.pairs[:, 0], weights=u[:, 0], minlength=SP.num_fish)
        u_y = np.bincount(sp.pairs[:, 0], weights=u[:, 1], minlength=SP.num_fish)
        U = np.column_stack((u_x, u_y))

        # this is should be better, but has not been tested enaugh
        #U = u.reshape((SP.num_fish, SP.num_fish, 2)).sum(axis=1)

        I_f = np.pi * params.fish_radius**2 * params.k_p / params.vel
        U = U * I_f / np.pi
        #I_f = 0.01

        pos_dx = np.column_stack((self.pos[:, 0] + delta, self.pos[:, 1]))
        sp_dx = self.get_spp_properties(pos_dx, self.dir)
        u_dx = self.flow_u(sp_dx, params)
        u_dx = (u_dx - u) / delta

        pos_dy = np.column_stack((self.pos[:, 0], self.pos[:, 1] + delta))
        sp_dy = self.get_spp_properties(pos_dy, self.dir)
        u_dy = self.flow_u(sp_dy, params)
        u_dy = (u_dy - u) / delta
        
        u_grad_ij = np.concatenate((u_dx[:,:, np.newaxis], u_dy[:,:, np.newaxis]), axis=2)
        u_grad_ij = u_grad_ij.transpose((0, 2, 1)) #(num_fish, dx-dy, dfun1-dfun2)

        u_grad_i = np.empty((SP.num_fish, 2, 2))
        for p in {(0,0), (0,1), (1,0), (1,1)}:
            u_grad_i[:, p[0], p[1]] = np.bincount(sp.pairs[:, 0], weights=u_grad_ij[:, p[0], p[1]], minlength=SP.num_fish)

        # this is should be better, but has not been tested enaugh
        #u_grad_i = u_grad_ij.reshape((SP.num_fish, SP.num_fish, 2, 2)).sum(axis=1)
        
        u_grad_i *= I_f / np.pi

        e_i = self.dir
        e_i_orth = np.column_stack((-e_i[:, 1], e_i[:, 0]))
        omega = (u_grad_i @ e_i_orth[:, :, np.newaxis])[:,:,0]
        omega = (e_i * omega).sum(axis=1)

        return U, omega
    
    def get_global_order_params(self, sp, params):
        P = np.linalg.norm(np.mean(self.dir, axis=0))

        center_of_mass = np.mean(self.pos, axis=0)

        e_i_r = (self.pos - center_of_mass) / np.linalg.norm(self.pos - center_of_mass, axis=1)[:, np.newaxis]

        r_i_dot = self.dir * params.vel

        V = np.mean(np.linalg.norm(r_i_dot, axis=0))

        # M = np.linalg.norm(np.mean(np.cross(e_i_r, r_i_dot), axis=0)) / np.linalg.norm(np.mean(e_i_r, axis=0)) * np.linalg.norm(np.mean(r_i_dot, axis=0))

        M = np.linalg.norm(np.mean(np.cross(e_i_r, r_i_dot), axis=0))

        if P < 0.5:
            if M < 0.4:
                phase = "SWARMING, \nP = " + str(round(P, 2)) + ", M = " + str(round(M, 2))
            else:
                phase = "MILLING, \nP = " + str(round(P, 2)) + ", M = " + str(round(M, 2))
        else:
            if M < 0.4:
                phase = "SCHOOLING, \nP = " + str(round(P, 2)) + ", M = " + str(round(M, 2))
            else:
                phase = "TURNING, \nP = " + str(round(P, 2)) + ", M = " + str(round(M, 2))

        self.phase = phase

    def calculate_distances(self):
        pos_rep = self.pos[:, :, np.newaxis].repeat(SP.num_fish, axis=2).transpose((2, 1, 0))
        dists = np.linalg.norm(pos_rep - self.pos[:,:, np.newaxis], axis=1)
        return dists
    
    def get_voronoi_neighbours(self):
        indptr_neigh, neighbours = Delaunay(self.pos).vertex_neighbor_vertices
        i_idx = np.repeat(np.arange(SP.num_fish), np.diff(indptr_neigh))
        idx = np.column_stack((i_idx, neighbours))
        return idx, idx[:, 0] * SP.num_fish + idx[:, 1]
    
    def resolve_collisions(self, params):
        min_dist = 2*params.fish_radius
        idx = np.argwhere(self.dists < min_dist)
        idx = idx[idx[:, 0] != idx[:, 1]]
        if len(idx) > 0:
            overlap = min_dist - self.dists[idx[:, 0], idx[:, 1]]
            direction = self.pos[idx[:, 0]] - self.pos[idx[:, 1]]
            norm = np.linalg.norm(direction, axis=1)
            norm = np.where(norm > 0, norm, 1)
            direction /= norm[:, np.newaxis]
            self.pos[idx[:, 0]] += direction * overlap[:, np.newaxis] * 0.5
import numpy as np
from simulation_parameters import SP
from scipy.spatial import Delaunay

np.random.seed(0)

class Response:
    def __init__(self, pos, dir, pred_pos, pred_dir):
        self.pos = pos
        self.dir = dir
        self.pred_pos = pred_pos
        self.pred_dir = pred_dir

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
        spawn_offset = 5

        # create initial positions and directions of all spp
        # (for updating the properties that are universal)
        self.spp_pos = np.random.uniform(spawn_offset, np.array(SP.aquarium_size)-spawn_offset, (SP.num_fish + SP.num_pred, 2))
        self.spp_dir = self.ang2dir(np.random.rand(SP.num_fish + SP.num_pred) * 2 * np.pi)

        # split spp into spp and predators
        self.pos = self.spp_pos.view()[:SP.num_fish]
        self.dir = self.spp_dir.view()[:SP.num_fish]

        self.pred_pos = self.spp_pos.view()[SP.num_fish:]
        self.pred_dir = self.spp_dir.view()[SP.num_fish:]
        
        # initialize distances between all fish
        self.dists = self.calculate_distances()

        self.phase = ""

    def ang2dir(self, alpha, out=None):
        if out is None:
            out = np.empty((len(alpha), 2))
        np.cos(alpha, out=out[:, 0])
        np.sin(alpha, out=out[:, 1])
        return out
    
    def dir2ang(self, dir, out=None):
        if out is None:
            out = np.empty((len(dir)))
        np.arctan2(dir[:, 1], dir[:, 0], out=out)
        return out

    def simulate(self, deltaTime, params):
        # calculate spp properties for all spp	
        sp = self.get_spp_properties(self.spp_pos, self.spp_pos, self.spp_dir, self.spp_dir)
        
        # FISH: consider alignment attraction factor for direction
        alignment_attraction = self.alignment_attraction(sp, params)

        # ALL: consider standard wiener process term with standard deviation params.sigma
        wiener = self.wiener(params)

        # ALL: consider flow factor for position offset
        flow_offset, omega = self.flow_offset(sp, params)

        # PRED: consider predator avoidance
        #avoidance, pred_attraction = self.predator_affect(sp, params)
        avoidance, pred_attraction = 0, 0


        # UPDATE
        # --------------------------------------------
        # DIR
        # --------------------------------------------
        # update all spp angles (randomness, flow)
        alpha = self.dir2ang(self.spp_dir)
        delta_alpha = np.zeros(SP.num_fish+SP.num_pred)

        # update all spp angle with wiener process and flow
        delta_alpha += wiener + omega
        #print(omega[:10])

        # update fish angle
        delta_alpha[:SP.num_fish] += alignment_attraction + avoidance
        
        # update predator angle
        delta_alpha[SP.num_fish:] += pred_attraction

        # consider delta time
        delta_alpha *= deltaTime

        alpha += delta_alpha

        # update pred direction
        self.ang2dir(alpha, out=self.spp_dir)

        # --------------------------------------------
        # POS
        # --------------------------------------------

        # offset as a result of momentum conservation
        dir = self.spp_dir.copy()
        dir[:SP.num_fish] *= params.vel
        dir[SP.num_fish:] *= params.pred_vel

        # update position
        self.spp_pos += (dir + flow_offset) * deltaTime

        # handle borders
        match params.borders:
            # Wrap around the aquarium
            case "loop":
                mask0 = self.spp_pos < 0
                mask1 = self.spp_pos > SP.aquarium_size
                if mask0.any():
                    self.spp_pos[:,0][mask0[:,0]] += SP.aquarium_size[0]
                    self.spp_pos[:,1][mask0[:,1]] += SP.aquarium_size[1]
                if mask1.any():
                    self.spp_pos[:,0][mask1[:,0]] -= SP.aquarium_size[0]
                    self.spp_pos[:,1][mask1[:,1]] -= SP.aquarium_size[1]
            # Bounce off the walls
            case "bounce":
                mask0 = self.spp_pos < 0
                mask1 = self.spp_pos > SP.aquarium_size
                if mask0.any():
                    self.spp_pos[mask0] = 0
                    self.spp_dir[mask0] = np.abs(self.spp_dir[mask0])
                if mask1.any():
                    self.spp_pos[:,0][mask1[:,0]] = SP.aquarium_size[0]
                    self.spp_pos[:,1][mask1[:,1]] = SP.aquarium_size[1]
                    self.spp_dir[mask1] = -np.abs(self.spp_dir[mask1])
            # Repel from the walls
            case "repulsion":
                mask1 = self.pos < 1/10*SP.aquarium_size[0]
                mask2 = self.pos > 9/10*SP.aquarium_size[0]
                self.dir[mask1] += (1/10*SP.aquarium_size[0] - self.pos[mask1]) * params.vel * deltaTime
                self.dir[mask2] -= (self.pos[mask2] - 9/10*SP.aquarium_size[0]) * params.vel * deltaTime
                self.dir /= np.linalg.norm(self.dir, axis=1)[:, np.newaxis]

                mask1 = self.pred_pos < 1/10*SP.aquarium_size[0]
                mask2 = self.pred_pos > 9/10*SP.aquarium_size[0]
                self.pred_dir[mask1] += (1/10*SP.aquarium_size[0] - self.pred_pos[mask1]) * params.pred_vel * deltaTime
                self.pred_dir[mask2] -= (self.pred_pos[mask2] - 9/10*SP.aquarium_size[0]) * params.pred_vel * deltaTime
                self.pred_dir /= np.linalg.norm(self.pred_dir, axis=1)[:, np.newaxis]

        # update distances between all fish
        self.dists = self.calculate_distances()

        # handle collisions
        if params.collisions:
            self.resolve_collisions(params)
        
        # update global order parameters
        self.get_global_order_params(sp, params)

        return Response(self.pos, self.dir, self.pred_pos, self.pred_dir)
    
    @staticmethod
    def get_spp_properties(pos0, pos1, dir0, dir1):
        # make pairs of all fish
        pairs = np.array(np.meshgrid(np.arange(len(pos0)), np.arange(len(pos1)))).T.reshape(-1, 2)

        e_ji = pos0[pairs[:, 0]] - pos1[pairs[:, 1]]
        dist = np.linalg.norm(e_ji, axis=1)
        dist = np.where(dist > 0, dist, 1)
        e_ji /= dist[:, np.newaxis]
        e_ji_orth = np.column_stack((-e_ji[:, 1], e_ji[:, 0]))

        e_i = dir0[pairs[:, 0]]
        e_j = dir1[pairs[:, 1]]

        cos = (-e_ji * e_i).sum(axis=1)
        cos = np.clip(cos, -1, 1)
        theta_ij = np.arccos(cos)
        theta_ij_sign = e_i * (-e_ji[:, [1, 0]])
        theta_ij_sign = np.sign(theta_ij_sign[:, 0] - theta_ij_sign[:, 1])
        theta_ij *= theta_ij_sign

        cos = (e_ji * e_i).sum(axis=1)
        cos = np.clip(cos, -1, 1)
        theta_ji = np.arccos(cos)
        theta_ji_sign = e_j * (e_ji[:, [1, 0]])
        theta_ji_sign = np.sign(theta_ji_sign[:, 0] - theta_ji_sign[:, 1])
        theta_ji *= theta_ji_sign

        cos = (e_i * e_j).sum(axis=1)
        cos = np.clip(cos, -1, 1)
        phi_ij = np.arccos(cos)
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
        I_n_f = params.sigma * np.power(params.vel*params.k_p, -1/4)
        I_n_p = params.sigma * np.power(params.pred_vel*params.k_p, -1/4)
        rand = np.random.normal(0, 1, (SP.num_fish + SP.num_pred))
        rand[:SP.num_fish] *= I_n_f
        rand[SP.num_fish:] *= I_n_p
        return rand
    
    def flow_u(self, sp, params):
        u = sp.e_ji_orth * np.sin(sp.theta_ji)[:, np.newaxis] + sp.e_ji * np.cos(sp.theta_ji)[:, np.newaxis]
        u = u / (sp.dist**2)[:, np.newaxis]
        #u[sp.dist <= 2*params.fish_radius] = 0
        return u

    def flow_offset(self, sp, params, delta=1e-6):
        # flow intensity constants
        I_f_f = np.pi * params.fish_radius**2 * params.k_p / params.vel
        I_f_p = np.pi * params.pred_radius**2 * params.k_p / params.pred_vel

        # flow velocity
        u = self.flow_u(sp, params)

        # derivative of flow velocity in x direction
        pos_dx = np.column_stack((self.spp_pos[:, 0] + delta, self.spp_pos[:, 1]))
        sp_dx = self.get_spp_properties(pos_dx, self.spp_pos, self.spp_dir, self.spp_dir)
        u_dx = self.flow_u(sp_dx, params)
        u_dx = (u_dx - u) / delta

        # derivative of flow velocity in y direction
        pos_dy = np.column_stack((self.spp_pos[:, 0], self.spp_pos[:, 1] + delta))
        sp_dy = self.get_spp_properties(pos_dy, self.spp_pos, self.spp_dir, self.spp_dir)
        u_dy = self.flow_u(sp_dy, params)
        u_dy = (u_dy - u) / delta
        
        # calculate gradient of flow velocity for every pair of fish
        u_grad_ij = np.concatenate((u_dx[:,:, np.newaxis], u_dy[:,:, np.newaxis]), axis=2)
        u_grad_ij = u_grad_ij.transpose((0, 2, 1)) #(num_fish^2, dx-dy, dfun1-dfun2)

        # calculate gradient of flow velocity
        u_grad_i = u_grad_ij.reshape((SP.num_fish+SP.num_pred, SP.num_fish+SP.num_pred, 2, 2))
        u_grad_i[:,:SP.num_fish] *= I_f_f / np.pi
        u_grad_i[:,SP.num_fish:] *= I_f_p / np.pi
        # a fish does not affect itself
        np.fill_diagonal(u_grad_i[:,:,0,0], 0)
        np.fill_diagonal(u_grad_i[:,:,0,1], 0)
        np.fill_diagonal(u_grad_i[:,:,1,0], 0)
        np.fill_diagonal(u_grad_i[:,:,1,1], 0)
        u_grad_i = u_grad_i.sum(axis=1)

        # calculate angular velocity
        e_i = self.spp_dir
        e_i_orth = np.column_stack((-e_i[:, 1], e_i[:, 0]))
        omega = (u_grad_i @ e_i_orth[:, :, np.newaxis])[:,:,0]
        omega = (e_i * omega).sum(axis=1)
        
        # compute flow position offset in place (reshape creates a view and not a copy)
        # puting this at the beginning of the function would break the code
        U = u.reshape((SP.num_fish+SP.num_pred, SP.num_fish+SP.num_pred, 2))
        U[:, :SP.num_fish] *= I_f_f / np.pi
        U[:, SP.num_fish:] *= I_f_p / np.pi
        # a fish does not affect itself
        np.fill_diagonal(U[:,:,0], 0)
        np.fill_diagonal(U[:,:,1], 0)
        U = U.sum(axis=1)

        return U, omega
    
    # turn away from the predator
    def predator_affect(self, sp, params):
        #--------------------------------------------
        # calculation of properties
        #--------------------------------------------
        e_ji = self.pos - self.pred_pos
        dist = np.linalg.norm(e_ji, axis=1)
        dist = np.where(dist > 0, dist, 1)
        e_ji /= dist[:, np.newaxis]
        e_ji_orth = np.column_stack((-e_ji[:, 1], e_ji[:, 0]))

        e_i = self.dir
        e_j = self.pred_dir

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

        theta_ij_star = np.where(theta_ij > 0, theta_ij - np.pi, theta_ij + np.pi)
        #--------------------------------------------

        # fish avoidance rotation
        avoidance = 1/dist * np.sin(theta_ij_star)
        weights = (1 + np.cos(theta_ij)) / 2 + 1e-6
        avoidance = params.pred_avoidance * avoidance * weights

        # predator attraction rotation
        pred_attraction = 1/dist * np.sin(theta_ji)
        pred_weights = (1 + np.cos(theta_ji)) + 1e-6
        pred_attraction = pred_attraction * pred_weights
        pred_attraction = pred_attraction.sum(axis=0) / pred_weights.sum(axis=0)
        pred_attraction = params.pred_attraction * pred_attraction

        return avoidance, pred_attraction

    
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
        pos_rep = self.spp_pos[:, :, np.newaxis].repeat(SP.num_fish+SP.num_pred, axis=2).transpose((2, 1, 0))
        dists = np.linalg.norm(pos_rep - self.spp_pos[:,:, np.newaxis], axis=1)
        return dists
    
    def get_voronoi_neighbours(self):
        indptr_neigh, neighbours = Delaunay(self.pos).vertex_neighbor_vertices
        i_idx = np.repeat(np.arange(SP.num_fish), np.diff(indptr_neigh))
        idx = np.column_stack((i_idx, neighbours))
        return idx, idx[:, 0] * (SP.num_fish + SP.num_pred) + idx[:, 1]
    
    def resolve_collisions(self, params):
        min_dist = np.empty_like(self.dists)
        # divide dists matrix into 4 regions (because predators have different radius)
        min_dist[:SP.num_fish, :SP.num_fish] = 2*params.fish_radius
        min_dist[:SP.num_fish, SP.num_fish:] = params.fish_radius + params.pred_radius
        min_dist[SP.num_fish:, :SP.num_fish] = params.fish_radius + params.pred_radius
        min_dist[SP.num_fish:, SP.num_fish:] = 2*params.pred_radius
        # get indices of all collisions
        idx = np.argwhere(self.dists < min_dist)
        idx = idx[idx[:, 0] != idx[:, 1]]
        if len(idx) > 0:
            overlap = min_dist[idx[:, 0], idx[:, 1]] - self.dists[idx[:, 0], idx[:, 1]]
            overlap = overlap[:, np.newaxis]

            direction = self.spp_pos[idx[:, 0]] - self.spp_pos[idx[:, 1]]
            norm = np.linalg.norm(direction, axis=1)
            norm = np.where(norm > 0, norm, 1)
            direction /= norm[:, np.newaxis]

            offset = direction * overlap

            x = np.bincount(idx[:, 0], weights=offset[:, 0], minlength=SP.num_fish+SP.num_pred)
            y = np.bincount(idx[:, 0], weights=offset[:, 1], minlength=SP.num_fish+SP.num_pred)
            direction = np.column_stack((x, y))
            self.spp_pos += direction * 0.5
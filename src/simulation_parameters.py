class StaticParameters:
    pixels_per_meter = 33
    aquarium_size = [30, 30]
    num_fish = 50
    fish_radius = 0.4

class DynamicParameters:
    def __init__(self,
    k_s,
    k_a,
    k_c,
    k_v,
    k_p,
    vel,
    collisions,
    borders,
    separation_distance,
    alignment_distance,
    cohesion_distance,
    flow_distance):
        self.k_s = k_s
        self.k_a = k_a
        self.k_c = k_c
        self.k_v = k_v
        self.k_p = k_p
        self.vel = vel
        self.collisions = collisions
        self.borders = borders
        self.separation_distance = separation_distance
        self.alignment_distance = alignment_distance
        self.cohesion_distance = cohesion_distance
        self.flow_distance = flow_distance

SP = StaticParameters()
DP = DynamicParameters(
    k_s = 0,
    k_a = 0,
    k_c = 0,
    k_v = 2.7,
    k_p = 0.41,
    vel = 1,
    collisions = True,
    borders="loop",
    separation_distance = 5*SP.fish_radius,
    alignment_distance = 20*SP.fish_radius,
    cohesion_distance = 15*SP.fish_radius,
    flow_distance = 10*SP.fish_radius
)
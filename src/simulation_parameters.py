class StaticParameters:
    pixels_per_meter = 15
    aquarium_size = [60, 60]
    num_fish = 100
    sigma = 3.1415/8

class DynamicParameters:
    def __init__(self,
    fish_radius,
    sigma,
    k_v,
    k_p,
    vel,
    collisions,
    borders):
        self.fish_radius = fish_radius
        self.sigma = sigma
        self.k_v = k_v
        self.k_p = k_p
        self.vel = vel
        self.collisions = collisions
        self.borders = borders

SP = StaticParameters()
DP = DynamicParameters(
    fish_radius = 0.05,
    sigma = 0.103,
    k_v = 0.826,
    k_p = 0.41,
    vel = 2,
    collisions = True,
    borders="loop"
)
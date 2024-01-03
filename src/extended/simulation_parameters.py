class StaticParameters:
    pixels_per_meter = 15
    aquarium_size = [60, 60]
    num_fish = 100
    num_pred = 2
    sigma = 3.1415/8

class DynamicParameters:
    def __init__(self,
    fish_radius,
    pred_radius,
    sigma,
    k_v,
    k_p,
    vel,
    pred_vel,
    pred_avoidance,
    pred_attraction,
    collisions,
    borders):
        self.fish_radius = fish_radius
        self.pred_radius = pred_radius
        self.sigma = sigma
        self.k_v = k_v
        self.k_p = k_p
        self.vel = vel
        self.pred_vel = pred_vel
        self.pred_avoidance = pred_avoidance
        self.pred_attraction = pred_attraction
        self.collisions = collisions
        self.borders = borders

SP = StaticParameters()
DP = DynamicParameters(
    fish_radius = 0.5,
    pred_radius = 1,
    sigma = 0.103,
    k_v = 0.826,
    k_p = 0.41,
    vel = 5,
    pred_vel= 5,
    pred_avoidance= 1,
    pred_attraction= 1,
    collisions = True,
    borders="loop"
)

class ColorPalette:
    # fish = [1, 42, 74, 255]
    fish = [245, 144, 37, 255]
    fish_alt = [230, 95, 50, 255]
    background = [202, 240, 248, 255]
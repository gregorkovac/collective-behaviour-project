class StaticParameters:
    pixels_per_meter = 15
    aquarium_size = [60, 60]
    num_fish = 100
    num_pred = 0
    sigma = 3.1415/8
    flow_field_size = 20

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
    borders,
    external_flow_angle,
    external_flow_mean,
    external_flow_amplitude,
    external_flow_velocity,
    external_flow_wavelength):
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
        self.external_flow_angle = external_flow_angle
        self.external_flow_mean = external_flow_mean
        self.external_flow_amplitude = external_flow_amplitude
        self.external_flow_velocity = external_flow_velocity
        self.external_flow_wavelength = external_flow_wavelength

SP = StaticParameters()
DP = DynamicParameters(
    fish_radius = 0.5,
    pred_radius = 1,
    sigma = 0.103,
    k_v = 0.826,
    k_p = 0.41,
    vel = 5,
    pred_vel= 5,
    pred_avoidance= 0.5,
    pred_attraction= 10,
    collisions = True,
    borders="loop",
    external_flow_angle = 0,
    external_flow_mean = 0.5,
    external_flow_amplitude= 1,
    external_flow_velocity = 20,
    external_flow_wavelength = 40
)

class ColorPalette:
    # fish = [1, 42, 74, 255]
    #fish = [50, 120, 80, 255]
    #fish_alt = [10, 75, 100, 255]
    fish = [245, 144, 37, 255]
    fish_alt = [230, 95, 50, 255]
    predator = [27, 20, 100, 255]
    predator_alt = [6, 82, 221, 255]
    predator_eyes = [255, 255, 255, 255]
    background = [10,5,40,255]#[202, 240, 248, 255]
    flow_hsv = [200, 50, 80]
    flow = [202,240,248,10]#[202, 240, 248, 255]
    flow_dir = [245, 144, 37, 60]
    flow_circle = [245, 144, 37, 60]#[20,60,120,100]#
    visualizations = [31, 198, 85, 255]
    boids = [230, 95, 50, 255]
    #background = [202/3, 240/4, 248/2, 255]
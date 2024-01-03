import dearpygui.dearpygui as dpg
from simulation_parameters import *

class GUI:
    def __init__(self):
        # map position in meters to pixels
        self.pos2pixels = lambda x: x * SP.pixels_per_meter

        # create gui elements
        self.create_gui()
        
        # add boids to the canvas
        self.boids = self.add_boids()
        self.dirs = self.add_dirs()

        self.pred_boid = self.add_pred_boid()
        self.pred_dir = self.add_pred_dir()
    
    @staticmethod
    def on_hover(sender, app_data):
        print("Hovered over {sender}")
        return
        dpg.configure_item(sender, alpha=1)  # fully opaque when hovered

    @staticmethod
    def on_unhover(sender, app_data):
        dpg.configure_item(sender, alpha=0.5)  # half transparent when not hovered

    def create_gui(self):
        dpg.create_context()
        
        #dpg.show_item_registry()
        dpg.create_viewport(
            title="Simulation",
            width=self.pos2pixels(SP.aquarium_size[0]),
            height=self.pos2pixels(SP.aquarium_size[1]),
            resizable=False)
        dpg.setup_dearpygui()

        
        self.canvas = dpg.add_viewport_drawlist(tag="Canvas", front=False)

        with dpg.window(label="Settings",
                        tag="Settings",
                        autosize=True,
                        no_title_bar=True,
                        pos=[10, 10]):
            with dpg.group(horizontal=True):
                dpg.add_text("Frame rate")
                dpg.add_text("0", tag="FPS", color=[0, 255, 0, 255])
            with dpg.group(horizontal=True):
                dpg.add_text("Phase")
                dpg.add_text("0", tag="phase", color=[0, 255, 0, 255])
            dpg.add_text("Simulation parameters")
            dpg.add_slider_float(
                label="sigma",
                tag="sigma",
                default_value=DP.sigma,
                min_value=0.0,
                max_value=10.0
            )
            dpg.add_slider_float(
                label="k_v",
                tag="k_v",
                default_value=DP.k_v,
                min_value=0.001,
                max_value=10.0
            )
            dpg.add_slider_float(
                label="k_p",
                tag="k_p",
                default_value=DP.k_p,
                min_value=0.001,
                max_value=10.0
            )
            dpg.add_checkbox(label="Collisions", tag="collisions", default_value=DP.collisions)
            dpg.add_combo(
                label="Borders",
                tag="borders",
                items=["loop", "bounce", "repulsion"],
                default_value=DP.borders
            )

            dpg.add_text("Fish parameters")
            dpg.add_slider_float(
                label="radius",
                tag="fish_radius",
                default_value=DP.fish_radius,
                min_value=0.0,
                max_value=2.0
            )
            dpg.add_slider_float(
                label="velocity",
                tag="vel",
                default_value=DP.vel,
                min_value=0.001,
                max_value=10.0
            )
            dpg.add_slider_float(
                label="pred_avoidance",
                tag="pred_avoidance",
                default_value=DP.pred_avoidance,
                min_value=0.001,
                max_value=10.0
            )

            dpg.add_text("Predator parameters")
            dpg.add_slider_float(
                label="radius",
                tag="pred_radius",
                default_value=DP.pred_radius,
                min_value=0.0,
                max_value=2.0
            )
            dpg.add_slider_float(
                label="velocity",
                tag="pred_vel",
                default_value=DP.pred_vel,
                min_value=0.001,
                max_value=10.0
            )
            dpg.add_slider_float(
                label="pred_attraction",
                tag="pred_attraction",
                default_value=DP.pred_attraction,
                min_value=0.001,
                max_value=10.0
            )

        # create a theme
        #theme_id = dpg.add_theme()

        # set the color of the window background
        #dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255, 128), parent=theme_id)  # half transparent white
        #dpg.set_item_theme("Settings", theme_id)


        dpg.show_viewport()

    def get_dynamic_parameters(self):
        # read dynamic parameteres from gui
        fish_radius = dpg.get_value("fish_radius")
        pred_radius = dpg.get_value("pred_radius")
        sigma = dpg.get_value("sigma")
        k_v = dpg.get_value("k_v")
        k_p = dpg.get_value("k_p")
        vel = dpg.get_value("vel")
        pred_vel = dpg.get_value("pred_vel")
        pred_avoidance = dpg.get_value("pred_avoidance")
        pred_attraction = dpg.get_value("pred_attraction")
        collisions = dpg.get_value("collisions")
        borders = dpg.get_value("borders")

        # create parameter object
        params = DynamicParameters(
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
        )
        return params
    
    def update_boids(self, res):
        pos = self.pos2pixels(res.pos)
        dir = self.pos2pixels(res.dir*2* dpg.get_value("fish_radius"))
        pred_pos = self.pos2pixels(res.pred_pos)
        pred_dir = self.pos2pixels(res.pred_dir*2* dpg.get_value("pred_radius"))

        # update gui
        for i in range(SP.num_fish):
            dpg.configure_item(item=self.boids[i], center=[pos[i, 0], pos[i, 1]], radius=self.pos2pixels(dpg.get_value("fish_radius")))
            dpg.configure_item(item=self.dirs[i],
                                p2=[pos[i, 0], pos[i, 1]],
                                p1=[pos[i, 0]-0.5*dir[i, 0], pos[i, 1]-0.5*dir[i, 1]], thickness=self.pos2pixels(dpg.get_value("fish_radius")))
        
        for i in range(SP.num_pred):
            dpg.configure_item(item=self.pred_boid[i], center=[pred_pos[i, 0], pred_pos[i, 1]], radius=self.pos2pixels(dpg.get_value("pred_radius")))
            dpg.configure_item(item=self.pred_dir[i],
                                p2=[pred_pos[i, 0], pred_pos[i, 1]],
                                p1=[pred_pos[i, 0]-0.5*pred_dir[i, 0], pred_pos[i, 1]-0.5*pred_dir[i, 1]], thickness=self.pos2pixels(dpg.get_value("pred_radius")))

    def update_frameRate(self, deltaTime):
        dpg.set_value("FPS", str(int(1/deltaTime)))

    def update_phase(self, phase):
        dpg.set_value("phase", str(phase))

    def add_pred_boid(self):
        boids = list()
        for i in range(SP.num_pred):
            boids.append(dpg.draw_circle(
                center=[0, 0],
                radius=self.pos2pixels(dpg.get_value("pred_radius")),
                color=[255, 0, 0, 255],
                fill=[255, 0, 0, 255],
                parent="Canvas",
            ))
        return boids
    
    def add_pred_dir(self):
        dirs = list()
        for i in range(SP.num_pred):
            dirs.append(dpg.draw_arrow(
                p1=[0, 0],
                p2=[0, 0],
                thickness=self.pos2pixels(dpg.get_value("pred_radius")),
                color=[255, 0, 0, 255],
                parent="Canvas",
            ))
        return dirs

    def add_boids(self):
        boids = list()
        for i in range(SP.num_fish):
            boids.append(dpg.draw_circle(
                center=[0, 0],
                radius=self.pos2pixels(dpg.get_value("fish_radius")),
                color=[0, 255, 0, 255],
                fill=[0, 255, 0, 255],
                parent="Canvas",
            ))
        return boids
    
    def add_dirs(self):
        dirs = list()
        for i in range(SP.num_fish):
            dirs.append(dpg.draw_arrow(
                p1=[0, 0],
                p2=[0, 0],
                thickness=self.pos2pixels(dpg.get_value("fish_radius")),
                color=[0, 255, 0, 255],
                parent="Canvas",
            ))
        return dirs
    
    def render_frame(self):
        dpg.render_dearpygui_frame()

    def is_running(self):
        return dpg.is_dearpygui_running()
    
    def destroy(self):
        dpg.destroy_context()
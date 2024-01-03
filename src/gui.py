import dearpygui.dearpygui as dpg
from simulation_parameters import *
import numpy as np

class GUI:
    def __init__(self):
        # map position in meters to pixels
        self.pos2pixels = lambda x: x * SP.pixels_per_meter

        # create gui elements
        self.create_gui()

        # set background color
        dpg.set_viewport_clear_color(color=ColorPalette.background)
        
        # add boids to the canvas
        self.tails = self.add_tails()
        self.dirs = self.add_dirs()
        self.dirs_2 = self.add_dirs_2()

        self.particles = self.add_boids()
        # self.boids = self.add_boids()

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
                        pos=[720, 740]):
            with dpg.group(horizontal=True):
                dpg.add_text("Frame rate")
                dpg.add_text("0", tag="FPS", color=ColorPalette.fish)
            with dpg.group(horizontal=True):
                dpg.add_text("Phase")
                dpg.add_text("0", tag="phase", color=ColorPalette.fish)
            dpg.add_text("Simulation parameters")
            dpg.add_slider_float(
                label="fish_radius",
                tag="fish_radius",
                default_value=DP.fish_radius,
                min_value=0.0,
                max_value=2.0
            )
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
            
            dpg.add_slider_float(
                label="vel",
                tag="vel",
                default_value=DP.vel,
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

        dpg.show_viewport()

    def get_dynamic_parameters(self):
        # read dynamic parameteres from gui
        fish_radius = dpg.get_value("fish_radius")
        sigma = dpg.get_value("sigma")
        k_v = dpg.get_value("k_v")
        k_p = dpg.get_value("k_p")
        vel = dpg.get_value("vel")
        collisions = dpg.get_value("collisions")
        borders = dpg.get_value("borders")

        # create parameter object
        params = DynamicParameters(
            fish_radius,
            sigma,
            k_v,
            k_p,
            vel,
            collisions,
            borders,
        )
        return params
    
    def update_boids(self, res):

        pos = self.pos2pixels(res.pos)
        dir = self.pos2pixels(res.dir*2* dpg.get_value("fish_radius"))

        # new_particles = self.add_boids()

        # update gui
        for i in range(SP.num_fish):
            # dpg.configure_item(item=self.boids[i], center=[pos[i, 0], pos[i, 1]], radius=self.pos2pixels(dpg.get_value("fish_radius")))
            dpg.configure_item(item=self.dirs[i],
                                p2=[pos[i, 0], pos[i, 1]],
                                p1=[pos[i, 0]-0.5*dir[i, 0], pos[i, 1]-0.5*dir[i, 1]], thickness=self.pos2pixels(dpg.get_value("fish_radius")))
            dpg.configure_item(item=self.dirs_2[i],
                                p2=[pos[i, 0], pos[i, 1]],
                                p1=[pos[i, 0]+0.5*dir[i, 0], pos[i, 1]+0.5*dir[i, 1]], thickness=self.pos2pixels(dpg.get_value("fish_radius")))
            dpg.configure_item(item=self.tails[i],
                                p2=[(pos[i, 0]-0.71*dir[i,0]), (pos[i, 1]-0.71*dir[i,1])],
                                p1=[pos[i, 0]-0.7*dir[i, 0], pos[i, 1]-0.7*dir[i, 1]], thickness=0.5*self.pos2pixels(dpg.get_value("fish_radius")))
            
        #     new_particles[i] = dpg.draw_circle(
        #         center=[pos[i, 0], pos[i, 1]],
        #         radius=0.1*self.pos2pixels(dpg.get_value("fish_radius")),
        #         color=ColorPalette.fish,
        #         fill=ColorPalette.fish,
        #         parent="Canvas",
        #     )

        # self.particles += new_particles

    def update_frameRate(self, deltaTime):
        dpg.set_value("FPS", str(int(1/deltaTime)))

    def update_phase(self, phase):
        dpg.set_value("phase", str(phase))

    def add_boids(self):
        boids = list()
        for i in range(SP.num_fish):
            boids.append(dpg.draw_circle(
                center=[0, 0],
                radius=self.pos2pixels(dpg.get_value("fish_radius")),
                color=ColorPalette.fish,
                fill=ColorPalette.fish,
                parent="Canvas",
            ))
        return boids
    
    def add_dirs(self):
        dirs = list()
        for i in range(SP.num_fish):
            dirs.append(dpg.draw_arrow(
                p1=[0, 0],
                p2=[100, 100],
                thickness=self.pos2pixels(dpg.get_value("fish_radius")),
                color=ColorPalette.fish,
                parent="Canvas",
            ))
        return dirs
    
    def add_dirs_2(self):
        dirs = list()
        for i in range(SP.num_fish):
            dirs.append(dpg.draw_arrow(
                p1=[0, 0],
                p2=[0, 0],
                thickness=self.pos2pixels(dpg.get_value("fish_radius")),
                color=ColorPalette.fish_alt,
                parent="Canvas",
            ))
        return dirs
    
    def add_tails(self):
        dirs = list()
        for i in range(SP.num_fish):
            dirs.append(dpg.draw_arrow(
                p1=[0, 0],
                p2=[0, 0],
                thickness=self.pos2pixels(dpg.get_value("fish_radius")),
                color=ColorPalette.fish,
                parent="Canvas",
            ))
        return dirs

    def render_frame(self):
        dpg.render_dearpygui_frame()

    def is_running(self):
        return dpg.is_dearpygui_running()
    
    def destroy(self):
        dpg.destroy_context()
import dearpygui.dearpygui as dpg
from simulation_parameters import *
import numpy as np
import colorsys

class GUI:
    def __init__(self):
        # map position in meters to pixels
        self.pos2pixels = lambda x: x * SP.pixels_per_meter

        # create gui elements
        self.create_gui()

        # set background color
        dpg.set_viewport_clear_color(color=ColorPalette.background)
        
        self.external_flow_field = self.add_flow_field()
        self.external_flow_field_arrows = self.add_flow_field_arrows()

        #self.flow_dir = self.add_flow_dir()
        
        # add boids to the canvas
        self.tails = self.add_tails()
        self.dirs = self.add_dirs()
        self.dirs_2 = self.add_dirs_2()

        self.pred_tails = self.add_pred_tails()
        self.pred_boids = self.add_pred_boid()
        self.pred_dir = self.add_pred_dir()
        self.pred_eyes = self.add_pred_eyes()

        dpg.draw_circle(
                center=[self.pos2pixels(SP.aquarium_size[0])/2, self.pos2pixels(SP.aquarium_size[1])/2],
                radius=self.pos2pixels(SP.aquarium_size[0]),
                color=[0, 100, 200, 100],
                fill=[0, 100, 200, 100],
                parent="Canvas",
            )
    
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
        settings_width = 400
        #dpg.show_item_registry()
        dpg.create_viewport(
            title="Simulation",
            width=self.pos2pixels(SP.aquarium_size[0])+settings_width,
            height=self.pos2pixels(SP.aquarium_size[1]),
            resizable=False)
        dpg.setup_dearpygui()

        
        self.canvas = dpg.add_viewport_drawlist(tag="Canvas", front=False)

        with dpg.window(label="Settings",
                        tag="Settings",
                        width=settings_width,
                        height=self.pos2pixels(SP.aquarium_size[1]),
                        #autosize=True,
                        no_move=True,
                        no_title_bar=True,
                        pos=[self.pos2pixels(SP.aquarium_size[0]), 0]):
            with dpg.group(horizontal=True):
                dpg.add_text("Frame rate")
                dpg.add_text("0", tag="FPS", color=ColorPalette.fish)
            with dpg.group(horizontal=True):
                dpg.add_text("Phase")
                dpg.add_text("0", tag="phase", color=ColorPalette.fish)
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

            dpg.add_text("External flow parameters")
            dpg.add_slider_float(
                label="flow_angle",
                tag="external_flow_angle",
                default_value=DP.external_flow_angle,
                min_value=-np.pi,
                max_value=np.pi
            )
            dpg.add_slider_float(
                label="flow_mean",
                tag="external_flow_mean",
                default_value=DP.external_flow_mean,
                min_value=0,
                max_value=10
            )
            dpg.add_slider_float(
                label="flow_amplitude",
                tag="external_flow_amplitude",
                default_value=DP.external_flow_amplitude,
                min_value=0,
                max_value=10
            )
            dpg.add_slider_float(
                label="flow_velocity",
                tag="external_flow_velocity",
                default_value=DP.external_flow_velocity,
                min_value=0,
                max_value=100
            )
            dpg.add_slider_float(
                label="flow_wavelength",
                tag="external_flow_wavelength",
                default_value=DP.external_flow_wavelength,
                min_value=0.01,
                max_value=100
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
        external_flow_angle = dpg.get_value("external_flow_angle")
        external_flow_mean = dpg.get_value("external_flow_mean")
        external_flow_amplitude = dpg.get_value("external_flow_amplitude")
        external_flow_velocity = dpg.get_value("external_flow_velocity")
        external_flow_wavelength = dpg.get_value("external_flow_wavelength")

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
            external_flow_angle,
            external_flow_mean,
            external_flow_amplitude,
            external_flow_velocity,
            external_flow_wavelength
        )
        return params
    
    def update_boids(self, res):
        pos = self.pos2pixels(res.pos)
        dir = self.pos2pixels(res.dir*2* dpg.get_value("fish_radius"))
        pred_pos = self.pos2pixels(res.pred_pos)
        pred_dir = self.pos2pixels(res.pred_dir*2* dpg.get_value("pred_radius"))
        flow_dir = self.pos2pixels(res.flow_dir*2* dpg.get_value("fish_radius"))
        external_flow_field = res.external_flow_field

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
            #dpg.configure_item(item=self.flow_dir[i],
            #                    p2=[pos[i, 0], pos[i, 1]],
            #                    p1=[pos[i, 0]+flow_dir[i, 0], pos[i, 1]+flow_dir[i, 1]], thickness=0.3*self.pos2pixels(dpg.get_value("fish_radius")))
                               
        
        for i in range(SP.num_pred):

            perp_dir = np.array([dir[i, 1], -dir[i, 0]])

            dpg.configure_item(item=self.pred_tails[i],
                                p2=[(pred_pos[i, 0]-0.51*pred_dir[i,0]), (pred_pos[i, 1]-0.51*pred_dir[i,1])],
                                p1=[pred_pos[i, 0]-0.5*pred_dir[i, 0], pred_pos[i, 1]-0.5*pred_dir[i, 1]], thickness=0.5*self.pos2pixels(dpg.get_value("pred_radius")))
            dpg.configure_item(item=self.pred_boids[i], center=[pred_pos[i, 0], pred_pos[i, 1]], radius=self.pos2pixels(dpg.get_value("pred_radius")))
            # dpg.configure_item(item=self.pred_dir[i],
            #                     p2=[pred_pos[i, 0], pred_pos[i, 1]],
            #                     p1=[pred_pos[i, 0]-0.5*pred_dir[i, 0], pred_pos[i, 1]-0.5*pred_dir[i, 1]], thickness=self.pos2pixels(dpg.get_value("pred_radius")))


        for i in range(SP.flow_field_size):
            for j in range(SP.flow_field_size):
                x = self.pos2pixels(i * 1/SP.flow_field_size * SP.aquarium_size[0] + 0.5 / SP.flow_field_size * SP.aquarium_size[0])
                y = self.pos2pixels(j * 1/SP.flow_field_size * SP.aquarium_size[1] + 0.5 / SP.flow_field_size * SP.aquarium_size[1])

                start = np.array([x, y])
                end = start + self.pos2pixels(external_flow_field[i, j])

                # dpg.configure_item(item=self.external_flow_field_arrows[i*SP.flow_field_size + j],
                #                     p2=start,
                #                     p1=end, thickness=self.pos2pixels(0.1))
                
                # mean = np.mean(external_flow_field[i, j])

                # if i == 0 and j == 0:
                #     print(np.linalg.norm(external_flow_field[i, j]))

                flow_color = self.get_flow_color(external_flow_field[i, j])
                dpg.configure_item(item=self.external_flow_field[i*SP.flow_field_size + j],
                                   color=flow_color, fill=flow_color)
                

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
                color=ColorPalette.predator,
                fill=ColorPalette.predator,
                parent="Canvas",
            ))
        return boids
    
    def add_flow_dir(self):
        dirs = list()
        for i in range(SP.num_fish):
            dirs.append(dpg.draw_arrow(
                p1=[0, 0],
                p2=[0, 0],
                thickness=self.pos2pixels(0.1),
                color=ColorPalette.visualizations,
                parent="Canvas",
            ))
        return dirs
    
    def add_pred_dir(self):
        dirs = list()
        for i in range(SP.num_pred):
            dirs.append(dpg.draw_arrow(
                p1=[0, 0],
                p2=[0, 0],
                thickness=self.pos2pixels(dpg.get_value("pred_radius")),
                color=ColorPalette.predator,
                parent="Canvas",
            ))
        return dirs
    
    def add_pred_dirs_2(self):
        dirs = list()
        for i in range(SP.num_pred):
            dirs.append(dpg.draw_arrow(
                p1=[0, 0],
                p2=[0, 0],
                thickness=self.pos2pixels(dpg.get_value("pred_radius")),
                color=ColorPalette.predator_alt,
                parent="Canvas",
            ))
        return dirs
    
    def add_pred_tails(self):
        dirs = list()
        for i in range(SP.num_pred):
            dirs.append(dpg.draw_arrow(
                p1=[0, 0],
                p2=[0, 0],
                thickness=self.pos2pixels(dpg.get_value("pred_radius")),
                color=ColorPalette.predator_alt,
                parent="Canvas",
            ))
        return dirs
    
    def add_pred_boids(self):
        boids = list()
        for i in range(SP.num_pred):
            boids.append(dpg.draw_circle(
                center=[0, 0],
                radius=self.pos2pixels(dpg.get_value("pred_radius")),
                color=ColorPalette.predator_alt,
                fill=ColorPalette.predator,
                parent="Canvas",
            ))
        return boids
    
    def add_pred_eyes(self):
        boids = list()
        for i in range(2*SP.num_pred):
            boids.append(dpg.draw_circle(
                center=[0, 0],
                radius=self.pos2pixels(dpg.get_value("pred_radius"))/2,
                color=ColorPalette.predator_eyes,
                fill=ColorPalette.predator_eyes,
                parent="Canvas",
            ))
        return boids

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
    
    def add_flow_field(self):
        dirs = list()
        for i in range(SP.flow_field_size):
            for j in range(SP.flow_field_size):
                x = self.pos2pixels(i * 1/SP.flow_field_size * SP.aquarium_size[0] + 0.5 / SP.flow_field_size * SP.aquarium_size[0])
                y = self.pos2pixels(j * 1/SP.flow_field_size * SP.aquarium_size[1] + 0.5 / SP.flow_field_size * SP.aquarium_size[1])

                # dirs.append(dpg.draw_arrow(
                #     p1=[x, y],
                #     p2=[x, y],
                #     thickness=self.pos2pixels(0.1),
                #     color=ColorPalette.visualizations,
                #     parent="Canvas",
                # ))

                dirs.append(dpg.draw_circle(
                    center = [x, y],
                    radius=2*self.pos2pixels(SP.aquarium_size[0]/(SP.flow_field_size*2)),
                    color=ColorPalette.flow,
                    fill=ColorPalette.flow,
                    parent="Canvas",
                ))
        return dirs
    
    def add_flow_field_arrows(self):
        dirs = list()
        for i in range(SP.flow_field_size):
            for j in range(SP.flow_field_size):
                x = self.pos2pixels(i * 1/SP.flow_field_size * SP.aquarium_size[0] + 0.5 / SP.flow_field_size * SP.aquarium_size[0])
                y = self.pos2pixels(j * 1/SP.flow_field_size * SP.aquarium_size[1] + 0.5 / SP.flow_field_size * SP.aquarium_size[1])

                dirs.append(dpg.draw_arrow(
                    p1=[x, y],
                    p2=[x, y],
                    thickness=self.pos2pixels(0.1),
                    color=ColorPalette.visualizations,
                    parent="Canvas",
                ))
        return dirs
    
    def get_flow_color(self, flow):
        # print(flow)

        flow_length = np.linalg.norm(flow)

        c = ColorPalette.flow_hsv.copy()

        c[1] += 10*flow_length

        if c[1] > 100:
            c[1] = 100
        elif c[1] < 0:
            c[1] = 0

        # c[2] += mean_flow * 50

        # c[0] += flow[0] * 50
        # c[2] += flow[1] * 50

        # if c[0] < 200:
        #     c[0] = 200
        # elif c[0] > 260:
        #     c[0] = 260

        # print(flow_length, " -> ", c)
        # if c[2] < 80:
        #     c[2] = 80
        # elif c[2] > 100:
        #     c[2] = 100

        c_rgb = colorsys.hsv_to_rgb(c[0]/360.0, c[1]/100.0, c[2]/100.0)

        return [c_rgb[0]*255, c_rgb[1]*255, c_rgb[2]*255, 255]


    def render_frame(self):
        dpg.render_dearpygui_frame()

    def is_running(self):
        return dpg.is_dearpygui_running()
    
    def destroy(self):
        dpg.destroy_context()
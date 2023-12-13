import numpy as np
import time
# import keyboard
import cv2 as cv
from simulation_properties import *
from scene_manager import SceneManager


class Scene:
    def __init__(self, screen_width, screen_height, frame_rate):
        # Screen properties
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.aspect_ratio = self.screen_width / self.screen_height
        self.aquarium_width = AQUARIUM_WIDTH
        self.aquarium_height = AQUARIUM_HEIGHT
        # Scene manager to update object in the scene
        self.scene_mng = SceneManager()

        # Image used to display simulation results each frame
        self.image_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # cv.namedWindow("Simulation", cv.WINDOW_NORMAL)
        # cv.imshow("Simulation", self.image_frame)

        # cv.createTrackbar("K_P", "Simulation", 0, 100, self.set_k_p)
        # cv.createTrackbar("K_V", "Simulation", 0, 100, self.set_k_v)


        # Time frame variables
        self.frame_rate = frame_rate
        self.desired_time_step = 1 / self.frame_rate
        self.start_time = 0.0
        self.delta_ime = 0.0

    # def set_k_p(self, value):
    #     self.scene_mng.k_p = value / 10.0

    # def set_k_v(self, value):
    #     self.scene_mng.k_v = value / 10.0

    def start_simulation(self):
        # Perform any initialization before main loop
        self.start_time = time.time()
        self.scene_mng.initialize()

        # Internal call for simulation loop
        self.simulation_loop()

    def simulation_loop(self):
        while True:
            # Update delta time
            self.delta_time = time.time() - self.start_time
            self.start_time = time.time()
            # print(1/self.delta_time, self.delta_time)


            # Check if a key is pressed to stop simulation
            #if keyboard.is_pressed('q'):
            #    print("Key 'q' pressed. Stopping simulation...")
            #    cv.destroyAllWindows()
            #    break

            # Update the scene at desired time step
            if True: #self.delta_time >= self.desired_time_step:
                # Perform all necessary calculations on the scene
                #self.delta_time = 1/60
                self.scene_mng.tick(self.delta_time)

                # Clear the image
                self.image_frame.fill(0)

                # Get fish locations and display them as pixels
                norm_fish_locations = self.scene_mng.get_fish_locations()
                self.draw_on_frame(norm_fish_locations)

                # Display the frame
                self.display_frame()

                # Finally reset start time for next frame
                #self.start_time = time.time()

    def display_frame(self):
        cv.imshow("Simulation", self.image_frame)
        # This is necessary otherwise cv does not display at all
        cv.waitKey(1)

    def valid_screen_position(self, screen_position):

        if screen_position[0] < 0 or screen_position[0] > self.screen_width or screen_position[1] < 0 or screen_position[1] > self.screen_height or np.isnan(screen_position[0]) or np.isnan(screen_position[1]):
            return False
        else:
            return True

    def draw_on_frame(self, locations):
        # Scale factor to convert from meters to pixels
        scale_x = self.screen_width / self.aquarium_width
        scale_y = self.screen_height / self.aquarium_height

        first = True
        for location in locations:
            # Convert real-world location (in meters) to screen coordinates (in pixels)
            screen_position = location * np.array([scale_x, scale_y])

            if not self.valid_screen_position(screen_position):
                continue

            if first == True:
                c = (0, 0, 255)
                first = False
            else:
                c = (0, 255, 0)

            # Draw the fish on the screen as circles
            cv.circle(self.image_frame, tuple(screen_position.astype(int)), 13, c, -1)


        # for l in self.scene_mng.debug_lines:
        #     line = l * np.array([scale_x, scale_y])

        #     cv.line(self.image_frame, tuple(line[0].astype(int)), tuple(line[1].astype(int)), (0, 0, 255), 1)

        first = True
        for d in self.scene_mng.main_dir:
            main_dir_vec = d * np.array([scale_x, scale_y])

            if not self.valid_screen_position(main_dir_vec[0]) or not self.valid_screen_position(main_dir_vec[1]):
                continue

            if first == True:
                c = (0, 0, 255)
                first = False
            else:
                c = (0, 255, 0)


            cv.arrowedLine(self.image_frame, main_dir_vec[0].astype(int), main_dir_vec[1].astype(int), c, 2)

        for f in self.scene_mng.flow:
            # print(f)

            flow_vec = f * np.array([scale_x, scale_y])

            if not self.valid_screen_position(flow_vec[0]) or not self.valid_screen_position(flow_vec[1]):
                continue

            cv.arrowedLine(self.image_frame, flow_vec[0].astype(int), flow_vec[1].astype(int), (0, 0, 255), 2)

        # cohesion_point = self.scene_mng.cohesion_point * np.array([scale_x, scale_y])

        # # print(self.scene_mng.cohesion_point)

        # cv.circle(self.image_frame, tuple(cohesion_point.astype(int)), 13, (0, 0, 255), -1)

        # main_dir_vec = self.scene_mng.main_dir * np.array([scale_x, scale_y])
        # cv.arrowedLine(self.image_frame, main_dir_vec[0].astype(int), main_dir_vec[1].astype(int), (0, 255, 0), 2)

        # for v in self.scene_mng.debug_dir:
        #     vec = v * np.array([scale_x, scale_y])

        #     print(vec)
        #     print("\n")

        #     cv.arrowedLine(self.image_frame, vec[0].astype(int), vec[1].astype(int), (255, 255, 255), 2)




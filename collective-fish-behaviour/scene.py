import numpy as np
import time
import keyboard
import cv2 as cv

from scene_manager import SceneManager


class Scene:
    def __init__(self, screen_width, screen_height, frame_rate):
        # Screen properties
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.aspect_ratio = self.screen_width / self.screen_height

        # Scene manager to update object in the scene
        self.scene_mng = SceneManager(self.aspect_ratio)

        # Image used to display simulation results each frame
        self.image_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # Time frame variables
        self.frame_rate = frame_rate
        self.desired_time_step = 1 / self.frame_rate
        self.start_time = 0.0
        self.delta_ime = 0.0

    def start_simulation(self):
        # Perform any initialization before main loop
        self.start_time = time.time()
        self.scene_mng.initialize()

        # Internal call for simulation loop
        self.simulation_loop()

    def simulation_loop(self):
        while True:
            # Update delta time
            self.delta_ime = time.time() - self.start_time

            # Check if a key is pressed to stop simulation
            if keyboard.is_pressed('q'):
                print("Key 'q' pressed. Stopping simulation...")
                cv.destroyAllWindows()
                break

            # Update the scene at desired time step
            if self.delta_ime >= self.desired_time_step:
                # Perform all necessary calculations on the scene
                self.scene_mng.tick(self.delta_ime)

                # Clear the image
                self.image_frame.fill(0)

                # Get fish locations and display them as pixels
                norm_fish_locations = self.scene_mng.get_fish_locations()
                self.draw_on_frame(norm_fish_locations)

                # Display the frame
                self.display_frame()

                # Finally reset start time for next frame
                self.start_time = time.time()

    def display_frame(self):
        cv.imshow("Simulation", self.image_frame)
        # This is necessary otherwise cv does not display at all
        cv.waitKey(1)

    def draw_on_frame(self, norm_locations):
        for norm_location in norm_locations:
            screen_position = norm_location * np.array([self.screen_width, self.screen_height])
            cv.circle(self.image_frame, tuple(map(int, screen_position)), 5, (0, 255, 0), -1)




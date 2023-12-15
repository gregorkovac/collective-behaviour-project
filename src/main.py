from gui import GUI
from simulation import Simulation
import time

class Engine:
    def __init__(self):
        self.gui = GUI()
        self.simulation = Simulation()
        self.previous_frame_time = 0
    
    def start(self):
        self.previous_frame_time = time.time()
        while self.gui.is_running():
            self.main_loop()
            self.gui.render_frame()
        self.gui.destroy()

    def main_loop(self):
        deltaTime = time.time() - self.previous_frame_time
        self.previous_frame_time = time.time()
        #deltaTime = 0.01

        params = self.gui.get_dynamic_parameters()
        res = self.simulation.simulate(deltaTime, params)

        self.gui.update_boids(res)
        self.gui.update_frameRate(deltaTime) 

def main():
    engine = Engine()
    engine.start()

if __name__ == '__main__':
    main()
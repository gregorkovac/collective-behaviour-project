from simulation_properties import *
from scene import Scene


def main():
    # Create Scene
    scene = Scene(SCREEN_WIDTH, SCREEN_HEIGHT, FRAME_RATE)

    # Start simulation
    scene.start_simulation()


if __name__ == "__main__":
    main()

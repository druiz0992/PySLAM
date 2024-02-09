from .landmark import Landmark
from .visualizer import Visualizer
from .grid import Grid

import numpy as np

class World:
    """
    Class for simulating the world

    Attributes:
    - world_grid: Grid representing the world
    - landmarks: List of landmarks in the world
    - visualizer: Visualizer for visualizing the world
    - x_max: Width of the world
    - y_max: Height of the world

    """

    def __init__(self, world_opts, visualizer_opts):
        """
        Initialize world with given dimensions.
        :param world_dims: Dimensions of the world [size_x, size_y]
        :param n_landmarks: Number of landmarks to add
        """

        self.x_max = world_opts['world_size'][0]
        self.y_max = world_opts['world_size'][1]

        self.landmarks = Landmark([self.x_max, self.y_max])
        self.landmarks.add_walls()

        # build world according to some predefined representation
        n_landmarks = world_opts['n_landmarks']
        if world_opts['world_type'] == 'reference_landmarks':
            self.landmarks.add_square_landmarks(n_landmarks, 1)
        elif world_opts['world_type'] == 'maze':
            self.landmarks.add_maze_landmarks()

        landmark_vertices = []
        for obj in self.landmarks_as_list():
            landmark_vertices.append(obj.vertices_as_array())

        # initialize visualizer with world and landmark vertices.
        self.visualizer = Visualizer([self.x_max, self.y_max], landmark_vertices, visualizer_opts)

        ## Create world grid
        self.world_grid = Grid(np.array(world_opts['world_size']) * np.array(world_opts['grid_scale']), world_opts['grid_scale'])
        self.world_grid.build(self.landmarks_as_list())

    def render(self, robot, particles, trajectory=None):
        """
        Draw the simulated world with its landmarks, the robot 2D pose and the particles that represent the discrete
        probability distribution that estimates the robot pose.

        :param robot: True robot 2D pose (x, y, heading)
        :param particles: Set of weighted particles (list of [weight, [x, y, heading]]-lists)
        """

        self.visualizer.render(robot, particles, trajectory)

    def grid(self):
        """
        Return the grid.
        :return: Grid
        """
        return self.world_grid

    def grid_as_array(self):
        """
        Return the grid.
        :return: Grid
        """
        return self.world_grid.as_array()


    def landmarks_as_list(self):
        """
        Return the landmarks in the world.
        :return: List of landmarks
        """
        return self.landmarks.landmarks_as_list()
    
    def size(self):
        """
        Return the world size.
        :return: World size
        """
        return [self.x_max, self.y_max]
    
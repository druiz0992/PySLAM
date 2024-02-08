from .landmark import Landmark
from .visualizer import Visualizer

class World:

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
        n_landmarks = world_opts['n_landmarks']
        if world_opts['world_type'] == 'reference_landmarks':
            self.landmarks.add_square_landmarks(n_landmarks, 1)
        elif world_opts['world_type'] == 'maze':
            self.landmarks.add_maze_landmarks()

        landmark_vertices = []
        for obj in self.landmarks.get_landmarks():
            landmark_vertices.append(obj.get_vertices())

        self.visualizer = Visualizer([self.x_max, self.y_max], landmark_vertices, visualizer_opts)

    def draw_world(self, robot, particles, trajectory=None):
        """
        Draw the simulated world with its landmarks, the robot 2D pose and the particles that represent the discrete
        probability distribution that estimates the robot pose.
        :param robot: True robot 2D pose (x, y, heading)
        :param particles: Set of weighted particles (list of [weight, [x, y, heading]]-lists)
        """

        self.visualizer.draw_world(robot, particles, trajectory)

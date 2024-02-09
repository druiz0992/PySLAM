import numpy as np
from .world_object import WorldObject

class Landmark:
    """
    Collection of WorldObjects representing landmarks in the world.

    Attributes:
    - landmarks: List of WorldObjects representing the landmarks
    - size_x: Width of the world
    - size_y: Height of the world
    """

    def __init__(self, world_dims):
        self.landmarks = []
        self.size_x = world_dims[0]
        self.size_y = world_dims[1]

    def add_landmark(self, landmark, check_intersection=True):
        """
        Add a landmark to the world. Raises an exception if the landmark intersects with another landmark.

        :param landmark: Landmark to add
        :param check_intersection: If True, check if the landmark intersects with any other landmark
        """
        assert isinstance(landmark, WorldObject)
        # Check if landmark collides with any other landmark
        if check_intersection:
            for other in self.landmarks:
                if landmark.intersects(other):
                    raise ValueError('Landmark intersects another landmark')
        self.landmarks.append(landmark)

    def landmarks_as_list(self):
        """
        Return the landmarks in the world.

        :return: List of landmarks
        """
        return self.landmarks

    def add_walls(self):
        """
        Add walls to the world.

        """
        check_intersection = False
        self.add_landmark(WorldObject.rectangle((self.size_x/2, 0.5), self.size_x, 1, 0), check_intersection)
        self.add_landmark(WorldObject.rectangle((self.size_x/2, self.size_y), self.size_x, 1, 0), check_intersection)
        self.add_landmark(WorldObject.rectangle((0, self.size_y/2), 1, self.size_y, 0), check_intersection)
        self.add_landmark(WorldObject.rectangle((self.size_x, self.size_y/2), 1, self.size_y, 0), check_intersection) 

    def add_maze_landmarks(self, p=[0.4, 0.6]):
        """
        Add maze-like rectangular landmarks to the world.

        :param p: Probability of a cell being occupied.
          First element is the probability of a cell being occupied,
          second element is the probability of a cell being free

        """
        # generate a maze
        maze = np.random.choice([1, 0], size=(int(self.size_y-1), int(self.size_x-1)), p=p)
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i, j] == 1:
                    landmark = WorldObject.rectangle((j, i), 1, 1, 0)
                    self.add_landmark(landmark, False)

    def add_square_landmarks(self, n_landmarks, size):
        """
        Add square landmarks to the world.

        :param n_landmarks: Number of landmarks to add
        :param size: Size of the landmarks
        """
        size_x = self.size_x
        size_y = self.size_y
        for _ in range(n_landmarks):
            landmark = WorldObject.square((np.random.uniform(size, size_x - size), np.random.uniform(size, size_y - size)), size, 0)
            # Add landmark to the world. Don't check for intersection, as the landmarks are randomly placed
            self.add_landmark(landmark, False)



    
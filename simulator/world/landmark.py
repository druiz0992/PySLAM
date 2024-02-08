import numpy as np
from .world_object import WorldObject

class Landmark:

    def __init__(self, world_dims):
        self.landmarks = []
        self.size_x = world_dims[0]
        self.size_y = world_dims[1]

    def add_landmark(self, landmark, check_intersection=True):
        """
        Add a landmark to the world.

        :param landmark: Landmark to add
        NOTE: landmark is an object of class WorldObject
        """
        assert isinstance(landmark, WorldObject)
        # Check if landmark collides with any other landmark
        if check_intersection:
            for other in self.landmarks:
                if landmark.intersects(other):
                    raise ValueError('Landmark intersects another landmark')
        self.landmarks.append(landmark)

    def get_landmarks(self):
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
        self.add_landmark(WorldObject.rectangle((self.size_x/2, 0), self.size_x, 1, 0), check_intersection)
        self.add_landmark(WorldObject.rectangle((self.size_x/2, self.size_y), self.size_x, 1, 0), check_intersection)
        self.add_landmark(WorldObject.rectangle((0, self.size_y/2), 1, self.size_y, 0), check_intersection)
        self.add_landmark(WorldObject.rectangle((self.size_x, self.size_y/2), 1, self.size_y, 0), check_intersection) 

    def add_maze_landmarks(self):
        """
        Add maze-like landmarks to the world.

        """
        maze = np.random.randint(2, size=(int(self.size_y-1), int(self.size_x-1)))
        # scale the maze to the world size
        # use maze tile and stack it to create a maze
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i, j] == 1:
                    landmark = WorldObject.rectangle((j, i), 1, 1, 0)
                    self.add_landmark(landmark)

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
            # add_landmark raises exception if collision. If collision, try again until no collision
            while True:
                try:
                    self.add_landmark(landmark)
                    break
                except ValueError:
                    landmark = WorldObject.square((np.random.uniform(size, size_x - size), np.random.uniform(size, size_y - size)), size, 0)



    
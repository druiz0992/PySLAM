import numpy as np
from .world_object import WorldObject
from .grid import Grid

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
        self.landmark_collection = {}
        self.size_x = world_dims[0]
        self.size_y = world_dims[1]
        self.landmark_grid = None

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

    def as_list(self):
        """
        Return the landmarks in the world.

        :return: List of landmarks
        """
        return self.landmarks
    
    def as_collection(self):
        """
        Return the landmarks in the world as a collection.

        :return: Collection of landmarks
        """
        return self.landmark_collection

    def as_grid_array(self):
        """
        Return the landmarks in the world as a grid.

        :return: Grid of landmarks
        """
        return self.landmark_grid.as_array()

    def as_grid(self):
        """
        Return the landmarks in the world as a grid.

        :return: Grid of landmarks
        """
        return self.landmark_grid

    def coordinates(self):
        """
        Return the coordinates of the landmarks in the world.

        :return: List of coordinates of the landmarks
        """
        return [value for value in self.landmark_collection.values()]

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

    def add_square_landmarks(self, n_landmarks, size, vertices = []):
        """
        Add square landmarks to the world.

        :param n_landmarks: Number of landmarks to add
        :param size: Size of the landmarks
        """
        size_x = self.size_x
        size_y = self.size_y
        for idx in range(n_landmarks):
            landmark = WorldObject.square((np.random.uniform(size, size_x - size), np.random.uniform(size, size_y - size)), size, 0)
            if len(vertices) > 0:
                landmark = WorldObject.square(vertices[idx], size, 0) 
            # Add landmark to the world. Don't check for intersection, as the landmarks are randomly placed
            self.add_landmark(landmark, False)

    def create_collection(self, landmarks_list, scale, remove_walls=False):
        """
        Create a collection of landmarks 

        """
        start_lm_idx = 0
        if remove_walls:
            start_lm_idx = 4

        scale_x, scale_y = scale
        grid = np.zeros((int(self.size_y*scale_y), int(self.size_x*scale_x)), dtype=np.int32)

        for idx, landmark in enumerate(landmarks_list[start_lm_idx:]):
            # add the landmark to the collection
            center = landmark.center()
            self.landmark_collection[idx+1] = center
            grid[np.arange(int(center[1]*scale_y), int((center[1]+1)*scale_y)), int(center[0]*scale_x):int((center[0]+1)*scale_x)] = idx+1

        self.landmark_grid = Grid.from_array(grid, scale)



    
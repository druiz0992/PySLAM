import numpy as np
from matplotlib.path import Path
import copy

class Grid:
    """
    A 2D grid composed of cells. Each cell can be occupied or free.
    A grid has a scale that allows to resample the grid to world coordinates.
    A higher scale means a finer grid.

    """

    # Offset to avoid numerical errors. When defining landmarks manually is it common
    # to have vertices that are exactly on the grid lines. This offset avoids numerical
    # errors when checking if a point is inside a landmark.
    SAMPLING_OFFSET = 1e-6

    def __init__(self, grid_size, scale):
        """
        Initialize the grid with given dimensions and scale.

        :param grid_size: Dimensions of the grid [size_x, size_y]
        :param scale: Scale of the grid [scale_x, scale_y]
        """
        self.grid_x = int(grid_size[0])
        self.grid_y = int(grid_size[1])
        self.scale_x = scale[0]
        self.scale_y = scale[1]
        self.grid = np.zeros((self.grid_y, self.grid_x))

    def copy(self):
        """
        Create a copy of the grid.

        :return: Copy of the grid
        """
        return copy.deepcopy(self.grid)
    
    @staticmethod
    def grid_from_array(array, scale):
        """
        Create a grid from an array.

        :param array: Array to create the grid from
        :return: Grid
        """
        # check if array is a numpy array
        if not isinstance(array, np.ndarray):
            raise ValueError('Array must be a numpy array')
        # check if array is 2D
        if len(array.shape) != 2:
            raise ValueError('Array must be 2D')

        grid = Grid([array.shape[1], array.shape[0]], scale)
        grid.grid = np.copy(array)
        grid.grid.flags.writeable = False
        return grid

    def build(self, landmarks):
        """
        Build the grid from landmarks.

        """
        # landmark can be a single element or a list
        if not isinstance(landmarks, list):
            landmarks = [landmarks]

        x, y = np.meshgrid(np.linspace(self.SAMPLING_OFFSET, self.grid_x/self.scale_x+self.SAMPLING_OFFSET, int(self.grid_x)), \
                            np.linspace(self.SAMPLING_OFFSET, self.grid_y/self.scale_y+self.SAMPLING_OFFSET , int(self.grid_y)))
        grid_points = np.vstack((x.flatten(), y.flatten())).T

        for landmark in landmarks:
            path = Path(landmark.vertices_as_array())
            mask = path.contains_points(grid_points)
            self.grid[mask.reshape(self.grid_y, self.grid_x)] = 1

        self.grid.flags.writeable = False
        return self.grid

    def is_occupied(self, point, world_coordinates=False):
        """
        Check if a point is occupied in the grid.

            _point = [int(round(point[0] * self.scale_x)), int(round(point[1] * self.scale_y))]
        :param point: 2D point to check
        :param world_coordinates: Boolean that indicates whether the point is in world coordinates
        :return: True if the point is occupied, False otherwise
        """
        # check point is a nx2 array
        _point = np.copy(np.asarray(point))
        if not isinstance(_point, np.ndarray):
            raise ValueError('Point must be a numpy array')
        if len(_point.shape) != 2 or _point.shape[1] != 2:
          _point = np.asarray([point])

        if world_coordinates:
            _point[:,0] = np.clip(np.round(_point[:,0] * self.scale_x), 0, self.grid_x-1)
            _point[:,1] = np.clip(np.round(_point[:,1] * self.scale_y), 0, self.grid_y-1)
        index = _point.astype(int)
        return self.grid[index[:,1], index[:,0]] > 0

    def as_array(self):
        """
        Return the grid.

        :return: Grid
        """
        return self.grid
    
    def size(self):
        """
        Return the grid size.

        :return: Grid size
        """
        return [self.grid_x, self.grid_y]
    
    def scale(self):
        """
        Return the scale.

        :return: Scale
        """
        return [self.scale_x, self.scale_y]
    

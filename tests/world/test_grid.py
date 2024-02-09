import unittest
import pytest
import numpy as np

from simulator.world import Grid, WorldObject

class TestGrid(unittest.TestCase):
    def test_init(self):
        grid = Grid([10, 10], [1, 1])
        self.assertTrue(np.array_equal(grid.as_array(), np.zeros((10, 10))))

    def test_init_dimension(self):
        grid = Grid([10, 20], [1, 1])
        self.assertTrue(np.array_equal(grid.as_array(), np.zeros((20, 10))))

    def test_init_scale(self):
        grid = Grid([10, 20], [2, 1])
        self.assertTrue(np.array_equal(grid.as_array(), np.zeros((20, 10))))

    def test_build_writ_eonly(self):
        grid = Grid([5, 5], [1, 1])
        center = [1, 1]
        size = 2
        rotation = 0
        landmark1 = WorldObject.square(center, size, rotation)
        grid.build([landmark1, landmark1])
        g = grid.as_array()
        with pytest.raises(Exception):
            g[0,0] = 1


    def test_build_grid(self):
        grid = Grid([5, 5], [1, 1])
        center1 = [1, 1]
        center2 = [2, 2]
        size = 2
        rotation = 0
        landmark1 = WorldObject.square(center1, size, rotation)
        landmark2 = WorldObject.square(center2, size, rotation)
        grid.build([landmark1, landmark2])
        self.assertTrue(np.array_equal(grid.as_array(), np.array([[1, 1, 0, 0, 0],
                                                                 [1, 1, 1, 0, 0],
                                                                 [0, 1, 1, 0, 0],
                                                                 [0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0]])))
    def test_build_grid_rotate(self):
        grid = Grid([5, 5], [1, 1])
        center1 = [1, 1]
        center2 = [2, 2]
        size = 4
        rotation = 0
        rotation2 = np.pi/4
        landmark1 = WorldObject.square(center1, size, rotation)
        landmark2 = WorldObject.square(center2, size, rotation2)
        grid.build([landmark1, landmark2])
        self.assertTrue(np.array_equal(grid.as_array(), np.array([[1, 1, 1, 0, 0],
                                                                 [1, 1, 1, 1, 0],
                                                                 [1, 1, 1, 1, 0],
                                                                 [0, 1, 1, 0, 0],
                                                                 [0, 0, 0, 0, 0]])))

    def test_build_grid_dimension(self):
        grid = Grid([5, 8], [1, 1])
        center1 = [1, 1]
        center2 = [2, 2]
        size = 2
        rotation = 0
        landmark1 = WorldObject.square(center1, size, rotation)
        landmark2 = WorldObject.square(center2, size, rotation)
        grid.build([landmark1, landmark2])
        self.assertTrue(np.array_equal(grid.as_array(), np.array([[1, 1, 0, 0, 0],
                                                                 [1, 1, 1, 0, 0],
                                                                 [0, 1, 1, 0, 0],
                                                                 [0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0]])))

    def test_build_scalex(self):
        grid = Grid([6, 6], [0.5, 1])
        center1 = [1, 1]
        center2 = [2, 2]
        size = 2
        rotation = 0
        landmark1 = WorldObject.square(center1, size, rotation)
        landmark2 = WorldObject.square(center2, size, rotation)
        grid.build([landmark1, landmark2])
        self.assertTrue(np.array_equal(grid.as_array(), np.array([[1, 0, 0, 0, 0,0],
                                                                 [1, 1, 0, 0, 0, 0],
                                                                 [0, 1, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0]])))

    def test_build_scaley(self):
        grid = Grid([6, 6], [1, 2])
        center1 = [1, 1]
        center2 = [2, 2]
        size = 2
        rotation = 0
        landmark1 = WorldObject.square(center1, size, rotation)
        landmark2 = WorldObject.square(center2, size, rotation)
        grid.build([landmark1, landmark2])
        self.assertTrue(np.array_equal(grid.as_array(), np.array([[1, 1, 0, 0, 0,0],
                                                                 [1, 1, 0, 0, 0, 0],
                                                                 [1, 1, 1, 0, 0, 0],
                                                                 [1, 1, 1, 0, 0, 0],
                                                                 [0, 1, 1, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0]])))

    def test_is_occupied(self):
        grid = Grid([5, 5], [2, 1])
        center = [1, 1]
        size = 2
        rotation = 0
        landmark = WorldObject.square(center, size, rotation)
        grid.build(landmark)
        self.assertTrue(grid.is_occupied([0, 0]))
        self.assertTrue(grid.is_occupied([1, 0]))
        self.assertTrue(grid.is_occupied([2, 0]))
        self.assertTrue(grid.is_occupied([3, 0]))
        self.assertTrue(grid.is_occupied([0, 1]))
        self.assertTrue(grid.is_occupied([1, 1]))
        self.assertTrue(grid.is_occupied([2, 1]))
        self.assertTrue(grid.is_occupied([3, 1]))
        self.assertFalse(grid.is_occupied([4, 0]))
        self.assertFalse(grid.is_occupied([4, 1]))
        self.assertFalse(grid.is_occupied([0, 2]))
        self.assertFalse(grid.is_occupied([1, 2]))
        self.assertFalse(grid.is_occupied([2, 2]))
        self.assertFalse(grid.is_occupied([3, 2]))
        self.assertFalse(grid.is_occupied([4, 2]))
        self.assertFalse(grid.is_occupied([0, 3]))
        self.assertFalse(grid.is_occupied([1, 3]))
        self.assertFalse(grid.is_occupied([2, 3]))
        self.assertFalse(grid.is_occupied([3, 3]))
        self.assertFalse(grid.is_occupied([4, 3]))
        self.assertFalse(grid.is_occupied([0, 4]))
        self.assertFalse(grid.is_occupied([1, 4]))
        self.assertFalse(grid.is_occupied([2, 4]))
        self.assertFalse(grid.is_occupied([3, 4]))
        self.assertFalse(grid.is_occupied([4, 4]))

if __name__ == '__main__':
    unittest.main()
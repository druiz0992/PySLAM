import unittest
import numpy as np

from simulator.world import WorldObject

class TestWorldObject(unittest.TestCase):
    def test_init(self):
        vertices = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        rotation = 0
        obj = WorldObject(vertices, rotation)
        self.assertTrue(np.array_equal(obj.vertices_as_array(), np.array(vertices)))

    def test_rotation(self):
        vertices = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        rotation = np.pi / 2
        obj = WorldObject(vertices, rotation)
        # rotate object should have same vertices as original object, but rotated
        self.assertTrue(set(map(tuple, obj.vertices_as_array().tolist())) == set(map(tuple,[[-1, 1], [1, 1], [1, -1], [-1, -1]])))

    def test_center(self):
        vertices = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        rotation = 0
        obj = WorldObject(vertices, rotation)
        self.assertTrue(np.array_equal(obj.center(), [0, 0]))

    def test_intersection(self):
        vertices = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        rotation = 0
        obj = WorldObject(vertices, rotation)
        self.assertTrue(obj.intersects(obj))

    def test_square(self):
        center = [0, 0]
        size = 2
        rotation = 0
        obj = WorldObject.square(center, size, rotation)
        self.assertTrue(set(map(tuple, obj.vertices_as_array().tolist())) == set(map(tuple,[[-1, 1], [1, 1], [1, -1], [-1, -1]])))

    def test_square_rot(self):
        center = [0, 0]
        size = 2
        rotation = np.pi / 4
        obj = WorldObject.square(center, size, rotation)
        self.assertTrue(np.sum(np.abs(obj.vertices_as_array() - np.array([[1.4142, 0.], [0, 1.4142], [-1.4142, 0], [0, -1.4142]]))) < 1e-2)

    def test_square_rot_tras(self):
        center = [1, 1]
        size = 2
        rotation = np.pi / 4
        obj = WorldObject.square(center, size, rotation)
        self.assertTrue(np.sum(np.abs(obj.vertices_as_array() - np.array([[2.4142, 1.], [1, 2.4142], [-0.4142, 1], [1, -0.4142]]))) < 1e-3)

    def test_rectangle(self):
        center = [0, 0]
        width = 1
        height = 2
        rotation = 0
        obj = WorldObject.rectangle(center, width, height, rotation)
        self.assertTrue(set(map(tuple, obj.vertices_as_array().tolist())) == set(map(tuple,[[0.5, 1], [0.5, -1], [-0.5, -1], [-0.5, 1]])))

    def test_rectangle_rot(self):
        center = [0, 0]
        width = 1
        height = 2
        rotation = np.pi / 2
        obj = WorldObject.rectangle(center, width, height, rotation)
        self.assertTrue(np.sum(np.abs(obj.vertices_as_array() - np.array([[1, -0.5], [1, 0.5], [-1., 0.5], [-1, -0.5]]))) < 1e-6)

    def test_rectangle_tras(self):
        center = [3, 3]
        width = 1
        height = 2
        rotation = 0
        obj = WorldObject.rectangle(center, width, height, rotation)
        self.assertTrue(np.sum(np.abs(obj.vertices_as_array() - np.array([[3.5, 4], [2.5, 4], [2.5, 2], [3.5, 2]]))) < 1e-6)

    def test_rectangle_rot_tras(self):
        center = [3, 3]
        width = 1
        height = 2
        rotation = np.pi / 2
        obj = WorldObject.rectangle(center, width, height, rotation)
        self.assertTrue(np.sum(np.abs(obj.vertices_as_array() - np.array([[4, 2.5], [4, 3.5], [2, 3.5], [2, 2.5]]))) < 1e-6)


    def test_triangle(self):
        center = [0, 0]
        size = 2
        rotation = 0
        obj = WorldObject.triangle(center, size, rotation)
        self.assertTrue(np.sum(np.abs(obj.vertices_as_array() - np.array([[0., 2.], [1.73205081, -1.], [-1.73205081, -1.]]))) < 1e-6)

    def test_triangle_rot(self):
        center = [0, 0]
        size = 2
        rotation = np.pi / 4
        obj = WorldObject.triangle(center, size, rotation)
        self.assertTrue(np.sum(np.abs(obj.vertices_as_array() - np.array([[1.4142, 1.4142], [0.5176, -1.9318], [-1.9318, 0.5176]]))) < 1e-3)

    def test_triangle_rot_tras(self):
        center = [1, 1]
        size = 2
        rotation = np.pi / 4
        obj = WorldObject.triangle(center, size, rotation)
        self.assertTrue(np.sum(np.abs(obj.vertices_as_array() - np.array([[2.4142, 2.4142], [1.5176, -0.9318], [-0.9318, 1.5176]]))) < 1e-3)

    def test_polygon(self):
        vertices = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        obj = WorldObject.polygon(vertices)
        self.assertTrue(np.array_equal(obj.vertices_as_array(), np.array(vertices)))

    def test_random_polygon(self):
        obj = WorldObject.random_polygon(4)
        self.assertEqual(obj.vertices_as_array().shape[0], 4)


if __name__ == '__main__':
    unittest.main()
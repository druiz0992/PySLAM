import unittest
import numpy as np

from simulator.world import WorldObject

class TestWorldObject(unittest.TestCase):
    def test_init(self):
        vertices = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        rotation = 0
        obj = WorldObject(vertices, rotation)
        self.assertTrue(np.array_equal(obj.get_vertices(), np.array(vertices)))

    def test_rotation(self):
        vertices = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        rotation = np.pi / 2
        obj = WorldObject(vertices, rotation)
        print(obj.get_vertices())
        # rotate object should have same vertices as original object, but rotated
        self.assertTrue(set(map(tuple, obj.get_vertices().tolist())) == set(map(tuple,[[-1, 1], [1, 1], [1, -1], [-1, -1]])))


    def test_get_center(self):
        vertices = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        rotation = 0
        obj = WorldObject(vertices, rotation)
        self.assertTrue(np.array_equal(obj.get_center(), [0, 0]))

    def test_collision(self):
        vertices = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        rotation = 0
        obj = WorldObject(vertices, rotation)
        self.assertTrue(obj.collision(obj))

    def test_square(self):
        center = [0, 0]
        size = 2
        rotation = 0
        obj = WorldObject.square(center, size, rotation)
        self.assertTrue(set(map(tuple, obj.get_vertices().tolist())) == set(map(tuple,[[-1, 1], [1, 1], [1, -1], [-1, -1]])))

    def test_circle(self):
        center = [0, 0]
        radius = 1
        obj = WorldObject.circle(center, radius)
        self.assertTrue(np.array_equal(obj.get_vertices(), np.array([1])))

    def test_rectangle(self):
        center = [0, 0]
        width = 1
        height = 2
        rotation = 0
        obj = WorldObject.rectangle(center, width, height, rotation)
        self.assertTrue(set(map(tuple, obj.get_vertices().tolist())) == set(map(tuple,[[0.5, 1], [0.5, -1], [-0.5, -1], [-0.5, 1]])))

    def test_triangle(self):
        center = [0, 0]
        size = 2
        rotation = 0
        obj = WorldObject.triangle(center, size, rotation)
        print(obj.get_vertices())
        self.assertTrue(np,sum(np.abs(obj.get_vertices() - np.array([[0., 2.], [1.73205081, -1.], [-1.73205081, -1.]]))) < 1e-6)

    def test_polygon(self):
        vertices = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        obj = WorldObject.polygon(vertices)
        self.assertTrue(np.array_equal(obj.get_vertices(), np.array(vertices)))

    def test_random_polygon(self):
        obj = WorldObject.random_polygon([0, 0], 4, 5)
        self.assertEqual(obj.get_vertices().shape[0], 5)



if __name__ == '__main__':
    unittest.main()
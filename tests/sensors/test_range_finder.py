
import unittest
import numpy as np

from simulator.sensors import RangeFinder
from simulator.world import Grid
import matplotlib.pyplot as plt

range_finder_params = {
    'resolution': 0.1,
    'std_noise': 0.1,
    'range': 1,
    'fov': 2,
    'type': 0,
    'z_hit': 0.8,
    'z_short': 0.1,
    'z_max': 0.05,
    'z_rand': 0.05
}

class WorldTest:
    def __init__(self, size, scale):
        self.world_size = size
        self.scale = scale
        grid = np.zeros((size[0] * scale[0], size[1] * scale[1]), dtype=np.int32)
        grid[int(size[0]/4*scale[0]):int(size[0]/4*3*scale[0]), int(size[1]/4*scale[1]):int(size[1]/4*3*scale[1])] = 1
        self.world_grid = Grid.from_array(grid, scale)
    
    def size(self):
        return self.world_size
    
    def grid_scale(self):
        return self.scale
    
    def grid(self):
        return self.world_grid
    
world = WorldTest([10, 10],[2,2])

class TestRangeFinder(unittest.TestCase):
    def test_init(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        rf = RangeFinder(world, params)
        rf.P, rf.footprint = rf.build_footprint([5,0,0],rf.fov, rf.range, rf.world.grid_scale(), form = rf.footprint_form)

        self.assertTrue(np.array_equal(rf.as_ones_array(), np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 1, 1, 1, 1, 1, 1, 1],
                                                                [0, 1, 1, 1, 1, 1, 1, 1],
                                                                [0, 1, 1, 1, 1, 1, 1, 1],
                                                                [0, 1, 1, 1, 1, 1, 1, 1],
                                                                [0, 0, 0, 0, 1, 0, 0, 0]])))
        
    def test_init_triangle(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        params['footprint_form'] = 'triangle'
        rf = RangeFinder(world, params)
        rf.P, rf.footprint = rf.build_footprint([5,0,0],rf.fov, rf.range, rf.world.grid_scale(), form = rf.footprint_form)
        self.assertTrue(np.array_equal(rf.as_ones_array(), np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 1, 1, 1, 1, 1, 1, 1],
                                                                [0, 1, 1, 1, 1, 1, 1, 1],
                                                                [0, 1, 1, 1, 1, 1, 1, 1],
                                                                [0, 1, 1, 1, 1, 1, 1, 1],
                                                                [0, 0, 0, 0, 0, 0, 0, 0]])))
        
    def test_init_rotate(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        rf = RangeFinder(world, params)
        rf.P, rf.footprint = rf.build_footprint([5,0,np.pi/4],rf.fov, rf.range, rf.world.grid_scale(), form = rf.footprint_form)

        self.assertTrue(np.array_equal(rf.as_ones_array(), np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                                                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                                                [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                                                                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                                                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])))
        
    def test_init_traslate(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        rf = RangeFinder(world, params)
        rf.P, rf.footprint = rf.build_footprint([5,5,0],rf.fov, rf.range, rf.world.grid_scale(), form = rf.footprint_form)

        self.assertTrue(np.array_equal(rf.as_ones_array(), np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0]])))

        
    def test_init_traslate_rot(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        rf = RangeFinder(world, params)
        rf.P, rf.footprint = rf.build_footprint([5,5,np.pi/4],rf.fov, rf.range, rf.world.grid_scale(), form = rf.footprint_form)
        self.assertTrue(np.array_equal(rf.as_ones_array(), np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                                                     [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                                                     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                                                     [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                                                                     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                                                     [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                     [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                                     [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])))

    def test_match(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        rf = RangeFinder(world, params)
        rf.P, rf.footprint = rf.build_footprint([5,0,0],rf.fov, rf.range, rf.world.grid_scale(), form = rf.footprint_form)
        match = rf.world_grid.cellwise_occupied(rf.footprint)
        self.assertTrue(np.array_equal(rf.origin(), np.array([6.0,0.0])))
        self.assertTrue(np.array_equal(match.as_ones_array(), np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                                                                        [0., 0., 0., 0., 0., 0., 0., 0.],
                                                                        [0., 0., 0., 0., 0., 0., 0., 0.],
                                                                        [0., 0., 0., 0., 0., 0., 0., 0.],
                                                                        [0., 0., 0., 0., 0., 0., 0., 0.],
                                                                        [0., 0., 0., 1., 1., 1., 0., 0.],
                                                                        [0., 0., 0., 1., 1., 1., 0., 0.],
                                                                        [0., 0., 0., 1., 1., 1., 0., 0.],
                                                                        [0., 0., 0., 1., 1., 1., 0., 0.],
                                                                        [0., 0., 0., 1., 1., 1., 0., 0.],
                                                                        [0., 0., 1., 1., 1., 1., 1., 0.],
                                                                        [0., 0., 1., 1., 1., 1., 1., 0.],
                                                                        [0., 0., 1., 1., 1., 1., 1., 0.],
                                                                        [0., 0., 1., 1., 1., 1., 1., 0.],
                                                                        [0., 0., 1., 1., 1., 1., 1., 0.],
                                                                        [0., 0., 0., 0., 0., 0., 0., 0.],
                                                                        [0., 0., 0., 0., 0., 0., 0., 0.],
                                                                        [0., 0., 0., 0., 0., 0., 0., 0.],
                                                                        [0., 0., 0., 0., 0., 0., 0., 0.],
                                                                        [0., 0., 0., 0., 0., 0., 0., 0.]])))

    def test_match_triangle(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        params['footprint_form'] = 'triangle'
        rf = RangeFinder(world, params)
        rf.P, rf.footprint = rf.build_footprint([5,0,0],rf.fov, rf.range, rf.world.grid_scale(), form = rf.footprint_form)
        match = rf.world_grid.cellwise_occupied(rf.footprint)
        self.assertTrue(np.array_equal(rf.origin(), np.array([6.0,0.0])))

        self.assertTrue(np.array_equal(match.as_ones_array(), np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 1, 1, 1, 1, 1, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0]])))
    def test_match_rotate(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        rf = RangeFinder(world, params)
        rf.P, rf.footprint = rf.build_footprint([5,0,np.pi/4],rf.fov, rf.range, rf.world.grid_scale(), form = rf.footprint_form)
        match = rf.world_grid.cellwise_occupied(rf.footprint)
        self.assertTrue(np.array_equal(rf.origin(), np.array([0.0,0.0])))

        self.assertTrue(np.array_equal(match.as_ones_array(), np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])))
    def test_match_traslate(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        rf = RangeFinder(world, params)
        rf.P, rf.footprint = rf.build_footprint([5,5,0],rf.fov, rf.range, rf.world.grid_scale(), form = rf.footprint_form)
        match = rf.world_grid.cellwise_occupied(rf.footprint)
        self.assertTrue(np.array_equal(rf.origin(), np.array([6.0,10.0])))

        self.assertTrue(np.array_equal(match.as_ones_array(), np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0]])))

        
    def test_match_traslate_rot(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        rf = RangeFinder(world, params)
        rf.P, rf.footprint = rf.build_footprint([5,5,np.pi/4],rf.fov, rf.range, rf.world.grid_scale(), form = rf.footprint_form)

        match = rf.world_grid.cellwise_occupied(rf.footprint)
        self.assertTrue(np.array_equal(rf.origin(), np.array([0.0,10.0])))

        self.assertTrue(np.array_equal(match.as_ones_array(), np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                                                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                                                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])))
 
    def test_sample_behavior(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        rf = RangeFinder(world, params)
        rf.P, rf.footprint = rf.build_footprint([5,0,0],rf.fov, rf.range, rf.world.grid_scale(), form = rf.footprint_form)
        for _ in range(1000):
            behavior, distance = rf._sample_sensor_model()
            if behavior == rf.Z_MAX_IDX: 
                self.assertTrue(distance == params['range'])
            elif behavior == rf.Z_RAND_IDX:
                self.assertTrue(distance <= params['range'])
            else:
                self.assertTrue(distance < params['range'])

    def test_measure(self):
        params = dict(range_finder_params)
        params['range'] = 10
        params['fov'] = np.pi/8
        rf = RangeFinder(world, params)

        d = np.zeros(10000, dtype=np.float32)
        for i in range(10000):
           d[i] = rf.measure([5,0,0])
        plt.hist(d, bins=100)
        plt.show()

        

if __name__ == '__main__':
    unittest.main()
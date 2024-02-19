
from .sensor import Sensor, SensorType
from ..utils import get_gaussian_noise_sample
from ..world import Grid
import matplotlib.path as mpath


from scipy.spatial.transform import Rotation as R
import numpy as np

class RangeFinder(Sensor):
    """
    Range finder sensor model

    Attributes:
    """
    Z_HIT_IDX = 0
    Z_SHORT_IDX = 1
    Z_RAND_IDX = 2
    Z_MAX_IDX = 3

    Z_SHORT_P = 0.1

    def __init__(self,world, opts):

        super().__init__(world)

        self.range = opts['range']
        self.resolution = opts['resolution']
        self.fov = opts['fov']
        self.std_noise = opts['std_noise']

        
        self.z = np.zeros(4, dtype=np.float32)
        self.z[self.Z_HIT_IDX] = opts['z_hit']
        self.z[self.Z_SHORT_IDX] = opts['z_short']
        self.z[self.Z_RAND_IDX] = opts['z_rand']
        self.z[self.Z_MAX_IDX] = opts['z_max']
        self.z_cum = np.cumsum(self.z)
        self.lambda_short = opts['lambda_short']


        if np.abs(np.sum(self.z) - 1) > 1e-6:
            raise ValueError('Sum of sensor model parameters must be 1')

        self.footprint_form = 'double-triangle'
        if 'footprint_form' in opts:
           self.footprint_form = opts['footprint_form']
        self.world_grid = world.grid()
        self.world = world
        self.P = None
        self.footprint = None


    def compute_likelihood(self, states, sensor_data):
        """
        """
        pass


    def sample_sensor_model(self, distances = [0]):
        """
        LiDAR sensor model
        """
        _shape = distances.shape
        _distances = distances.reshape(-1)
        z = np.random.rand(len(_distances))
        result = np.ones(len(_distances), dtype=np.float32) * self.range + self.resolution
        behavior = np.ones(len(_distances), dtype=np.int32) * self.Z_RAND_IDX

        out_of_range_mask = _distances > self.range
        # Check conditions for each sample
        # HIT samples out of reach are left as they are
        # HIT sample within reach are updated with a gaussian noise sample
        hit_mask = z <= self.z_cum[self.Z_HIT_IDX] 
        result[hit_mask] = \
            get_gaussian_noise_sample(_distances[hit_mask], self.std_noise)
        result[(hit_mask) & (out_of_range_mask)] = self.range + self.resolution
        behavior[hit_mask] = self.Z_HIT_IDX

        # all SHORT samples are updated with an exponential noise sample, but clipped to the sensor's range
        short_mask = (z > self.z_cum[self.Z_HIT_IDX]) & (z <= self.z_cum[self.Z_SHORT_IDX]) 
        #n = 1/ (1 - np.exp(-self.lambda_short * _distances[short_mask]))
        result[short_mask] = np.random.exponential(self.lambda_short, np.count_nonzero(short_mask)) * _distances[short_mask]
        result[short_mask] = np.clip(result[short_mask], 0, _distances[short_mask]) 
        behavior[short_mask] = self.Z_SHORT_IDX


        # all RANDOM samples are updated with a uniform noise sample, but clipped to the sensor's range
        rand_mask = (z > self.z_cum[self.Z_SHORT_IDX]) & (z <= self.z_cum[self.Z_RAND_IDX]) 
        result[rand_mask] = np.random.rand(np.count_nonzero(rand_mask)) * self.range
        behavior[short_mask] = self.Z_RAND_IDX

        max_mask = z > self.z_cum[self.Z_RAND_IDX]
        result[max_mask] = self.range + self.resolution
        behavior[max_mask] = self.Z_MAX_IDX

        result = np.clip(result, 0, self.range + self.resolution)

        result = result.reshape(_shape)
        behavior = behavior.reshape(_shape)
        distances.reshape(_shape)

        return (result, behavior)


    def calibrate(self):
        pass


@staticmethod
def model(grid, robot_pose, sensor_pose, sensor_params):
    """
    """
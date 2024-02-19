
from .sensor import Sensor, SensorType
from ..utils import get_gaussian_noise_sample

class Gyroscope(Sensor):
    """
    Gyroscope class

    Gyroscope measurement model is given by the following equation:
    w = w_real + w_bias + w_noise

    w_real is the real angular velocity of the robot and its computed with this equation:
    w_real = change_in_angle / change_in_time, and can be computed based on change in robot's pose
      change_in_angle = current_angle - previous_angle = theta_at_t - theta_at_t-1

    w_bias is the bias of the gyroscope and it's a constant value that is added to the real angular velocity

    w_noise is the noise of the gyroscope and it's a random value that is added to the real angular velocity

    where:
    """

    def __init__(self,world, pose, opts):
        super().__init__(world, pose, SensorType.GYROSCOPE)

        self.std_bias = opts['std_bias'] # measured in rad/hr, and modelled as a normal distribution
        self.mean_bias = opts['mean_bias'] # measured in rad/hr
        self.std_noise = opts['std_noise'] # measured in rad/sqrt(hr) and modelled as a normal distribution

    def measure(self, angle_difference, time_difference):
        """
        Measure the angular velocity of the robot based on the robot's pose

        :param angle_difference: change in angle of the robot
        :param time_difference: change in time between the two angles
        return: noisy angular velocity of the robot
        """
        w = angle_difference / time_difference
        w += get_gaussian_noise_sample(0, self.std_noise*((time_difference/3600)**0.5))
        w += get_gaussian_noise_sample(self.mean_bias * time_difference/3600, self.std_bias*((time_difference/3600)**0.5))

    def calibrate(self):
        pass


@staticmethod
def model(grid, robot_pose, sensor_pose, sensor_params):
    """
    Model the gyroscope sensor

    :param grid: world grid
    :param robot_pose: robot pose
    :param sensor_pose: sensor pose
    :param sensor_params: sensor parameters
    :return: noisy angular velocity of the robot
    """

    # Get the angle difference between the current and previous pose
    angle_difference = robot_pose[2] - sensor_pose[2]

    # Get the time difference between the current and previous pose
    time_difference = robot_pose[3] - sensor_pose[3]

    # Create a gyroscope sensor
    gyro = Gyroscope(grid, sensor_pose, sensor_params)

    # Measure the angular velocity of the robot
    w = gyro.measure(angle_difference, time_difference)

    return w

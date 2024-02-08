import numpy as np

from .robot import *


class TwoWheeledRobot(Robot):

    def __init__(self, robot_pose, robot_params, robot_noise_params):
        """
        Initialize the robot with given 2D pose. In addition set motion uncertainty parameters.

        :param x: Initial robot x-position (m)
        :param y: Initial robot y-position (m)
        :param theta: Initial robot heading (rad)
        :param robot_params: Dictionary with robot parameters
        :param robot_noise_params: Dictionary with robot noise parameters
        """
        # Unpack robot noise parameters
        std_meas_distance = robot_noise_params['std_meas_distance']
        std_meas_angle = robot_noise_params['std_meas_angle']

        self.std_rpm = robot_noise_params['std_rpm']

        ## Initialize fixed robot noise parameters
        self.std_wheelL_diameter = self._get_gaussian_noise_sample(0, robot_noise_params['std_wheel_diameter'])[0]
        self.std_wheelR_diameter = self._get_gaussian_noise_sample(0, robot_noise_params['std_wheel_diameter'])[0]
        self.std_wheel_distance = self._get_gaussian_noise_sample(0, robot_noise_params['std_wheel_distance'])[0]

        # Unpack robot parameters
        self.wheel_distance = robot_params['wheel_distance'] 
        self.wheelL_diameter = robot_params['wheel_diameter'] 
        self.wheelR_diameter = robot_params['wheel_diameter'] 
        self.skip_angle_measurement = robot_params['skip_angle_measurement']

        self.timestamp = 0.0

        self.x = self.x_raw = self.x_est = robot_pose[0]
        self.y = self.y_raw = self.y_est =  robot_pose[1]
        self.theta = self.theta_raw = self.theta_est = robot_pose[2]

        # Set standard deviation measurements
        self.std_meas_distance = std_meas_distance
        self.std_meas_angle = std_meas_angle

        self.trajectory = []
        self.trajectory.append([self.x, self.y, self.theta, self.x_raw, self.y_raw, self.theta_raw, self.timestamp, self.x_est, self.y_est, self.theta_est])


    def get_trajectory(self):
        return np.array(self.trajectory)

    def move(self, desired_motion, world, timestamp):
        """
        Move the robot according to given arguments and within world of given dimensions. The true motion is the sum of
        the desired motion and additive Gaussian noise that represents the fact that the desired motion cannot exactly
        be realized, e.g., due to imperfect control and sensing.

        ----------------------------------------------------------------------------------------------------------------

        :param desired_motion: left right motor rpm
        :param world: dimensions of the cyclic world in which the robot executes its motion
        """
       
        desired_Lrpm, desired_Rrpm = desired_motion

        # Compute actual motion including noise
        Lrpm = desired_Lrpm + self._get_gaussian_noise_sample(0, self.std_rpm)[0]
        Rrpm = desired_Rrpm + self._get_gaussian_noise_sample(0, self.std_rpm)[0]

        minutes_lapse = (timestamp - self.timestamp) / 60.0 
        Ldistance = (Lrpm * np.pi * (self.wheelL_diameter + self.std_wheelL_diameter)) * minutes_lapse 
        Rdistance = (Rrpm * np.pi * (self.wheelR_diameter + self.std_wheelR_diameter)) * minutes_lapse

        distance_driven = (Ldistance + Rdistance) / 2

        # Update robot pose
        self.x += distance_driven * np.cos(self.theta)
        self.y += distance_driven * np.sin(self.theta)
        self.theta += (Rdistance - Ldistance) / (self.wheel_distance + self.std_wheel_distance)

        # Check world limits
        # TODO: check collision with landmarks
        self.x = max(min(self.x, world.x_max-1), 1)
        self.y = max(min(self.y, world.y_max-1), 1)
        self.theta = np.mod(self.theta, 2*np.pi)

        # Raw motion: 
        Ldistance_raw = (desired_Lrpm * np.pi * self.wheelL_diameter) * minutes_lapse
        Rdistance_raw = (desired_Rrpm * np.pi * self.wheelR_diameter) * minutes_lapse
        distance_driven_raw = (Ldistance_raw + Rdistance_raw) / 2

        self.x_raw += distance_driven_raw * np.cos(self.theta_raw)
        self.y_raw += distance_driven_raw * np.sin(self.theta_raw)
        self.x_raw = max(min(self.x_raw, world.x_max-1), 1)
        self.y_raw = max(min(self.y_raw, world.y_max-1), 1)
        self.theta_raw += (Rdistance_raw - Ldistance_raw) / self.wheel_distance
        self.theta_raw = np.mod(self.theta_raw, 2*np.pi)

        self.trajectory.append([self.x, self.y, self.theta, self.x_raw, self.y_raw, self.theta_raw, timestamp])

        self.timestamp = timestamp
      
    def get_raw_pose(self):
        return [self.x_raw, self.y_raw, self.theta_raw]
    
    def set_initial_pose(self, pose):
        self.x = pose[0]
        self.y = pose[1]
        self.theta = pose[2]
        self.x_raw = pose[0]
        self.y_raw = pose[1]
        self.theta_raw = pose[2]
        self.trajectory[-1][3:6] = pose
        self.trajectory[-1][7:10] = pose

    def get_pose(self):
        return [self.x, self.y, self.theta]
    
    def get_estimated_pose(self):
        return [self.x_est, self.y_est, self.theta_est]
    
    def get_prev_raw_pose(self):
        return self.trajectory[-2][3:6]
  
    def get_raw_incremental_movement(self):
        return [np.sqrt((self.x_raw - self.trajectory[-2][3])**2 + (self.y_raw - self.trajectory[-2][4])**2), self.theta_raw - self.trajectory[-2][5]]
    
    def measure_error(self):
        error = 0.0
        for sample in self.trajectory:
            error += (sample[0] - sample[7])**2 + (sample[1] - sample[8])**2 + (sample[2] - sample[9])**2
        error =  np.sqrt(error) / len(self.trajectory)
        return error


    def set_estimated_pose(self, pose):
        """
        Estimate the robot pose based on the given particles. The estimate is the mean of the particle positions.

        :param particles: List of particles.
        """
        self.x_est = pose[0]
        self.y_est = pose[1]
        self.theta_est = pose[2]
        # append estimated pose to last trajectory sample
        self.trajectory[-1].extend([self.x_est, self.y_est, self.theta_est])


    def measure(self, landmarks_coordinates):
        """
        Perform a measurement. The robot is assumed to measure the distance to and angles with respect to all landmarks
        in meters and radians respectively. While doing so, the robot experiences zero mean additive Gaussian noise.

        :param world: World containing the landmark positions.
        :return: List of lists: [[dist_to_landmark1, angle_wrt_landmark1], dist_to_landmark2, angle_wrt_landmark2], ...]
        """
        skip_angle_measurement = self.skip_angle_measurement
        # Loop over measurements
        landmarks = np.array(landmarks_coordinates)

        # these landmarks' measurements are in the robot's frame of reference, which means that
        #  the robot performs some measurements and computes the distance and angle to each line of 
        #  sight landmark
        dx = self.x - landmarks[:, 0]
        dy = self.y - landmarks[:, 1]

        distances = np.sqrt(dx**2 + dy**2)
        z_distances = self._get_gaussian_noise_sample(distances, self.std_meas_distance)

        z_angles = np.zeros_like(distances)
        if not skip_angle_measurement:
           angles = np.arctan2(dy, dx)
           z_angles = self._get_gaussian_noise_sample(angles, self.std_meas_angle)
     
        measurements = np.array([z_distances, z_angles]).T

        return measurements

    @staticmethod
    def _get_gaussian_noise_sample(mu, sigma):
        """
        Get a random sample from a 1D Gaussian distribution with mean mu and standard deviation sigma.

        :param mu: mean of distribution
        :param sigma: standard deviation
        :return: random sample from distribution with given parameters
        """
        size = 1 if np.isscalar(mu) else mu.shape
        return np.random.normal(loc=mu, scale=sigma, size=size)

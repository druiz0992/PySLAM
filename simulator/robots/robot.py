import numpy as np
from abc import ABC, abstractmethod

class Robot(ABC):

    X_POSE_IDX = 0
    Y_POSE_IDX = 1
    THETA_POSE_IDX = 2

    TRAJECTORY_POSE_IDX = 0
    TRAJECTORY_RAW_POSE_IDX = 3
    TRAJECTORY_TIMESTAMP_IDX = 6
    TRAJECTORY_ESTIMATED_POSE_IDX = 7

    def __init__(self, robot_pose, robot_params, robot_noise_params):
        """
        Initialize the robot with given 2D pose. In addition set motion uncertainty parameters.

        :param robot_pose: 2D pose of the robot [x, y, theta]
        :param robot_params: Dictionary with robot parameters
        :param robot_noise_params: Dictionary with robot noise parameters

        """

        # Unpack robot noise parameters
        std_meas_distance = robot_noise_params['std_meas_distance']
        std_meas_angle = robot_noise_params['std_meas_angle']

        self.timestamp = 0.0


        self.x = self.x_raw = self.x_est = robot_pose[self.X_POSE_IDX]
        self.y = self.y_raw = self.y_est =  robot_pose[self.Y_POSE_IDX]
        self.theta = self.theta_raw = self.theta_est = robot_pose[self.THETA_POSE_IDX]

        # Set standard deviation measurements
        self.std_meas_distance = std_meas_distance
        self.std_meas_angle = std_meas_angle


        self.trajectory = []
        self.trajectory.append([self.x, self.y, self.theta, self.x_raw, self.y_raw, self.theta_raw, self.timestamp, self.x_est, self.y_est, self.theta_est])

        # World grid
        self.world_grid = None
        self.x_world_size, self.y_world_size = robot_params['world_size']

    def get_trajectory(self):
        return np.array(self.trajectory)
    
    def get_raw_pose(self):
        return [self.x_raw, self.y_raw, self.theta_raw]
    
    def set_initial_pose(self, pose):
        self.x = pose[self.X_POSE_IDX]
        self.y = pose[self.Y_POSE_IDX]
        self.theta = pose[self.THETA_POSE_IDX]
        self.x_raw = pose[self.X_POSE_IDX]
        self.y_raw = pose[self.Y_POSE_IDX]
        self.theta_raw = pose[self.THETA_POSE_IDX]
        self.trajectory[-1][self.TRAJECTORY_RAW_POSE_IDX:self.TRAJECTORY_TIMESTAMP_IDX] = pose
        self.trajectory[-1][self.TRAJECTORY_ESTIMATED_POSE_IDX:] = pose

    def get_pose(self):
        return np.array([self.x, self.y, self.theta])
    
    def get_estimated_pose(self):
        return [self.x_est, self.y_est, self.theta_est]
    
    def get_prev_raw_pose(self):
        return self.trajectory[-2][self.TRAJECTORY_RAW_POSE_IDX:self.TRAJECTORY_TIMESTAMP_IDX]
  
    def get_raw_incremental_movement(self):
        return [np.sqrt(
                  (self.x_raw - self.trajectory[-2][self.TRAJECTORY_RAW_POSE_IDX])**2 + \
                  (self.y_raw - self.trajectory[-2][self.TRAJECTORY_RAW_POSE_IDX+self.Y_POSE_IDX])**2
                  ),
                self.theta_raw - self.trajectory[-2][self.TRAJECTORY_RAW_POSE_IDX+self.THETA_POSE_IDX]]
    
    def measure_error(self):
        error = 0.0
        for sample in self.trajectory:
            error += \
                (sample[self.TRAJECTORY_POSE_IDX + self.X_POSE_IDX] -
                     sample[self.TRAJECTORY_ESTIMATED_POSE_IDX + self.X_POSE_IDX])**2 + \
                (sample[self.TRAJECTORY_POSE_IDX + self.Y_POSE_IDX] - 
                     sample[self.TRAJECTORY_ESTIMATED_POSE_IDX + self.Y_POSE_IDX])**2 + \
                (sample[self.TRAJECTORY_POSE_IDX + self.THETA_POSE_IDX] - 
                     sample[self.TRAJECTORY_ESTIMATED_POSE_IDX + self.THETA_POSE_IDX])**2
        error =  np.sqrt(error) / len(self.trajectory)
        return error

    def set_estimated_pose(self, pose):
        """
        Estimate the robot pose based on the given particles. The estimate is the mean of the particle positions.

        :param particles: List of particles.
        """
        self.x_est = pose[self.X_POSE_IDX]
        self.y_est = pose[self.Y_POSE_IDX]
        self.theta_est = pose[self.THETA_POSE_IDX]
        # append estimated pose to last trajectory sample
        self.trajectory[-1].extend([self.x_est, self.y_est, self.theta_est])


    def initialize_grid(self, grid, update_pose_if_collision=False):
        """
        Set the grid representation of the world in which the robot operates. The grid is used to ensure that the robot
        does not move through walls.

        :param grid: Grid representation of the world
        :param update_pose_if_collision: Boolean that indicates whether the robot pose should be updated if a collision
                                         is detected
        """
        self.world_grid = grid

        if update_pose_if_collision:
            self.reset_pose()
    
    def reset_pose(self):
        """
        Reset the robot pose to a random position in the world that is not occupied by a wall.
        """
        # check that the robot is not inside a landmark. If it is, select a new starting pose and check again
        while self.world_grid.is_occupied(self.get_pose()[:self.THETA_POSE_IDX], world_coordinates=True):
            self.set_initial_pose([np.random.uniform(1, self.x_world_size-2), \
                                    np.random.uniform(1, self.y_world_size-2), np.random.uniform(0, 2 * np.pi)])
    @abstractmethod
    def move(self, desired_motion, timestamp):
        """
        Move the robot according to given arguments and within world of given dimensions. The true motion is the sum of
        the desired motion and additive Gaussian noise that represents the fact that the desired motion cannot exactly
        be realized, e.g., due to imperfect control and sensing.

        :param desired_motion: Motion parameters
        :param timestamp: Time stamp of the motion

        """
        pass


    @abstractmethod
    def measure(self, landmarks_coordinates):
        """
        Perform a measurement. The robot is assumed to measure the distance to and angles with respect to all landmarks
        in meters and radians respectively. While doing so, the robot experiences zero mean additive Gaussian noise.

        :param landmarks_coordinates: List of landmarks' coordinates
        :return: List of measurements [distance, angle]
        """
        pass

    
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

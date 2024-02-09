import numpy as np
from abc import ABC, abstractmethod

class Trajectory:
    """
    Trajectory class. Stores robot trajectory.
        self.trajectory.append([self.x, self.y, self.theta, self.x_raw, self.y_raw, self.theta_raw, self.timestamp, self.x_est, self.y_est, self.theta_est])
    """
    TRAJECTORY_POSE_IDX = 0
    TRAJECTORY_RAW_POSE_IDX = 3
    TRAJECTORY_TIMESTAMP_IDX = 6
    TRAJECTORY_ESTIMATED_POSE_IDX = 7

    def __init__(self):
        self.trajectory = []

    def clear(self):
        self.trajectory = []

    def store(self, real_pose, raw_pose, estimated_pose, timestamp):
        if estimated_pose is not None:
            self.trajectory.append([*real_pose, *raw_pose, timestamp, *estimated_pose])
        else:
            self.trajectory.append([*real_pose, *raw_pose, timestamp])
        print(self.trajectory[-1])
        input()

    def store_estimated_pose(self, estimated_pose):
        self.trajectory[-1].extend(estimated_pose)
        print("ESTORE STIMATED")
        print(self.trajectory[-1])
        input()
    
    def get_trajectory(self):
        return np.array(self.trajectory)
    
    def get_last_two(self):
        return self.trajectory[-2:]
    
    def length(self):
        return len(self.trajectory)
    

class Robot(ABC):

    X_POSE_IDX = 0
    Y_POSE_IDX = 1
    THETA_POSE_IDX = 2


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


        self.trajectory = Trajectory()
        print("ROBOT INIT")
        self.trajectory.store(
            [self.x, self.y, self.theta],
            [self.x_raw, self.y_raw, self.theta_raw],
            [self.x_est, self.y_est, self.theta_est],
            self.timestamp
        )

        # World grid
        self.world_grid = None
        self.x_world_size, self.y_world_size = robot_params['world_size']

    def get_trajectory(self):
        return self.trajectory.get_trajectory()
    
    def get_raw_pose(self):
        return [self.x_raw, self.y_raw, self.theta_raw]
    
    def set_initial_pose(self, pose):
        self.x = pose[self.X_POSE_IDX]
        self.y = pose[self.Y_POSE_IDX]
        self.theta = pose[self.THETA_POSE_IDX]
        self.x_raw = pose[self.X_POSE_IDX]
        self.y_raw = pose[self.Y_POSE_IDX]
        self.theta_raw = pose[self.THETA_POSE_IDX]

        self.trajectory.clear()
        print("INIT POSE")
        self.trajectory.store(
            [self.x, self.y, self.theta],
            [self.x_raw, self.y_raw, self.theta_raw],
            [self.x_est, self.y_est, self.theta_est],
            self.timestamp
        )

    def get_pose(self):
        return np.array([self.x, self.y, self.theta])
    
    def get_estimated_pose(self):
        return [self.x_est, self.y_est, self.theta_est]
    
    def get_raw_incremental_movement(self):
        trajectory = self.trajectory.get_last_two()
        try:
           return [np.sqrt(
                  (self.x_raw - trajectory[Trajectory.TRAJECTORY_RAW_POSE_IDX])**2 + \
                  (self.y_raw - trajectory[Trajectory.TRAJECTORY_RAW_POSE_IDX+self.Y_POSE_IDX])**2
                  ),
                self.theta_raw - trajectory[Trajectory.TRAJECTORY_RAW_POSE_IDX+self.THETA_POSE_IDX]]
        except:
            return np.array([0., 0.])
    
    def measure_error(self):
        error = 0.0
        for sample in self.trajectory.get_trajectory():
            error += \
                (sample[Trajectory.TRAJECTORY_POSE_IDX + self.X_POSE_IDX] -
                     sample[Trajectory.TRAJECTORY_ESTIMATED_POSE_IDX + self.X_POSE_IDX])**2 + \
                (sample[Trajectory.TRAJECTORY_POSE_IDX + self.Y_POSE_IDX] - 
                     sample[Trajectory.TRAJECTORY_ESTIMATED_POSE_IDX + self.Y_POSE_IDX])**2 + \
                (sample[Trajectory.TRAJECTORY_POSE_IDX + self.THETA_POSE_IDX] - 
                     sample[Trajectory.TRAJECTORY_ESTIMATED_POSE_IDX + self.THETA_POSE_IDX])**2
        error =  np.sqrt(error) / self.trajectory.length()
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
        self.trajectory.store_estimated_pose([self.x_est, self.y_est, self.theta_est])


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

    
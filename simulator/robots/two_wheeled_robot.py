import numpy as np

from .robot import *


class TwoWheeledRobot(Robot):

    def __init__(self, robot_pose, robot_params, robot_noise_params):
        """
        Initialize the robot with given 2D pose. In addition set motion uncertainty parameters.

        :param robot_pose: 2D pose of the robot [x, y, theta]
        :param robot_params: Dictionary with robot parameters
        :param robot_noise_params: Dictionary with robot noise parameters

        """
        super().__init__(robot_pose, robot_params, robot_noise_params)

        self.std_rpm = robot_noise_params['std_rpm']

        ## Initialize fixed robot noise parameters
        self.std_wheelL_diameter = self._get_gaussian_noise_sample(0, robot_noise_params['std_wheel_diameter'])[0]
        self.std_wheelR_diameter = self._get_gaussian_noise_sample(0, robot_noise_params['std_wheel_diameter'])[0]
        self.std_wheel_distance = self._get_gaussian_noise_sample(0, robot_noise_params['std_wheel_distance'])[0]

        # Unpack robot paiameters
        self.wheel_distance = robot_params['wheel_distance'] 
        self.wheelL_diameter = robot_params['wheel_diameter'] 
        self.wheelR_diameter = robot_params['wheel_diameter'] 
        self.skip_angle_measurement = robot_params['skip_angle_measurement']


    def move(self, desired_motion, timestamp):
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

        attempt_x = self.x + distance_driven * np.cos(self.theta)
        attempt_y = self.y + distance_driven * np.sin(self.theta)
        self.theta += (Rdistance - Ldistance) / (self.wheel_distance + self.std_wheel_distance)
        self.theta = np.mod(self.theta, 2*np.pi)

        if not self.world_grid.is_occupied([attempt_x, attempt_y], world_coordinates=True):
             # Update robot pose
            self.x = attempt_x
            self.y = attempt_y

        # Raw motion: This is the motion that the robot would have executed if there were no noise.
        # For ease of reprensentation, we limit the robot's motion to the world's dimensions, but the
        # raw motion doesnt observe obstacles
        Ldistance_raw = (desired_Lrpm * np.pi * self.wheelL_diameter) * minutes_lapse
        Rdistance_raw = (desired_Rrpm * np.pi * self.wheelR_diameter) * minutes_lapse
        distance_driven_raw = (Ldistance_raw + Rdistance_raw) / 2

        self.x_raw += distance_driven_raw * np.cos(self.theta_raw)
        self.y_raw += distance_driven_raw * np.sin(self.theta_raw)
        self.x_raw = max(min(self.x_raw, self.x_world_size-1), 1)
        self.y_raw = max(min(self.y_raw, self.y_world_size-1), 1)
        self.theta_raw += (Rdistance_raw - Ldistance_raw) / self.wheel_distance
        self.theta_raw = np.mod(self.theta_raw, 2*np.pi)

        # store trajectory
        self.trajectory.append([self.x, self.y, self.theta, self.x_raw, self.y_raw, self.theta_raw, timestamp])

        self.timestamp = timestamp


    def measure(self, landmarks_coordinates):
        """
        Perform a measurement. The robot is assumed to measure the distance to and angles with respect to all landmarks
        in meters and radians respectively. While doing so, the robot experiences zero mean additive Gaussian noise.

        :param landmarks_coordinates: List of landmarks' coordinates
        :return: List of measurements [distance, angle]

        """
        skip_angle_measurement = self.skip_angle_measurement
        # Loop over measurements
        landmarks = np.array(landmarks_coordinates)

        # these landmarks' measurements are in the robot's frame of reference, which means that
        #  the robot performs some measurements and computes the distance and angle to each line of 
        #  sight landmark
        dx = self.x - landmarks[:, self.X_POSE_IDX]
        dy = self.y - landmarks[:, self.Y_POSE_IDX]

        distances = np.sqrt(dx**2 + dy**2)
        z_distances = self._get_gaussian_noise_sample(distances, self.std_meas_distance)

        z_angles = np.zeros_like(distances)
        if not skip_angle_measurement:
           angles = np.arctan2(dy, dx)
           z_angles = self._get_gaussian_noise_sample(angles, self.std_meas_angle)
     
        measurements = np.array([z_distances, z_angles]).T

        return measurements
    


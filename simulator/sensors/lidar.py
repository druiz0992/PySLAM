
from ..sensors import RangeFinder

import numpy as np

class Lidar(RangeFinder):
    """
    Lidar

    """

    def __init__(self,world, opts):
        super().__init__(world, opts)

        self.n_scans = opts['n_scans']
        self.beam_offset = opts['beam_offset']
        self.n_beams = opts['n_beams']
        self.beam_separation = 2 * np.pi / self.n_beams
        self.beam_detectors = self.beam_offset + np.arange(self.n_beams) * self.beam_separation
        

    def scan(self, pose, add_noise = True):
        """
        Scan the environment and return LiDAR sensor readings

        :param pose: robot's pose
        :param add_noise: flag to add noise to the sensor readings
        """
        # array of sensor readings
        scanned_vector = np.ones(self.n_beams, dtype=np.float32) * self.range + self.resolution
        # array of landmark indices
        scanned_landmarks = np.ones(self.n_beams, dtype=np.int32) * -1
        landmarks = np.array(self.world.landmark_coordinates())

        # measure distance and angle to landmarks
        distances_sq = np.sum((landmarks - pose[:2])**2, axis=1)
        angles = np.mod(np.arctan2(landmarks[:,1] - pose[1], landmarks[:,0] - pose[0]) - pose[2], 2*np.pi)

        # find the detector that is closest to the estimated angle, and within the sensor's field of view
        diff = np.abs(angles[:,None] - self.beam_detectors)
        # indicates which landmark is detected by which detector. 
        closest_detector_idx = np.argmin(diff, axis=1)
        closest_detector_mask = np.abs(angles - self.beam_detectors[closest_detector_idx]) < self.fov

        detector_mask = np.abs(angles[:,None] - self.beam_detectors) < self.fov
        detected_landmarks = np.where(detector_mask)[0]
        detector = np.where(detector_mask)[1]
        
        inside_cirlce = distances_sq < self.range**2
        reachable_mask = (inside_cirlce) & (closest_detector_mask)

        # retrive direct line of sight reachable landmarks. Direct line of sight is the closest reachable landmark
        #  for a given angle
        for lm_idx in range(len(landmarks)):
            if not reachable_mask[lm_idx]:
                continue
            distance = np.sqrt(distances_sq[lm_idx])
            detected_lm_mask = detected_landmarks == lm_idx
            detector_beams = detector[detected_lm_mask]
            detection_mask = scanned_vector[detector_beams] > distance
            scanned_vector[detector_beams[detection_mask]] = distance
            scanned_landmarks[detector_beams[detection_mask]] = lm_idx

        # sampled distances is the distance measured to a line of sight landmark by one of the detectors
        # the angle can be extracted from the beam that detected the landmark
        sampled_distance = scanned_vector
        raw_distances = np.zeros(self.n_beams, dtype=np.float32)

        if add_noise:
           for _ in range(self.n_scans):
               m, _ = self.sample_sensor_model(sampled_distance)
               raw_distances += m
           
           sampled_distance = raw_distances / self.n_scans

        return (sampled_distance, scanned_landmarks)

    
    def compute_likelihood(self, states, sensor_data, measurement_noise):
        """
        LiDAR likelihood computation for particle filter

        :param states: particles
        :param sensor_data: sensor readings
        :param measurement_noise: measurement noise

        """
        if not isinstance(sensor_data, tuple):
            raise ValueError('Invalid sensor data')

        if not isinstance(states, np.ndarray):
            raise ValueError('Invalid states')

        sensor_distances, sensor_landmarks = sensor_data
        if not isinstance(sensor_distances, np.ndarray) or not isinstance(sensor_landmarks, np.ndarray):
            raise ValueError('Invalid sensor data')
        
        if len(sensor_distances) != len(sensor_landmarks) != self.n_beams:
            raise ValueError('Invalid sensor data')

        likelihood = np.zeros(len(states), dtype=np.float32)

        sensor_distances_inv = self.range + self.resolution - sensor_distances
        
        for state_idx, state in enumerate(states):
            raw_estimation, _= self.scan(state, False)
            #likelihood[state_idx] = self._compute_likelihood(raw_estimation, estimated_landmarks, sensor_distances, sensor_landmarks, measurement_noise)
            likelihood[state_idx] = self._best_match(raw_estimation, sensor_distances_inv, measurement_noise, state)

        return likelihood
    
    def _compute_likelihood(self, raw_estimation, estimated_landmarks, sensor_distances, sensor_landmarks, measurement_noise):
        """
        Likelihood computation for a given particle and if landmarks are known
        """
        likelihood = 1
        for lm in set(sensor_landmarks[sensor_landmarks >= 0]):
            idx = np.where( sensor_landmarks == lm)
            avg_distance = np.mean(sensor_distances[idx[0]])
            avg_angle = np.mod(np.mean(idx[0]) * self.beam_separation, 2*np.pi)

            ridx = np.where(estimated_landmarks == lm) 
            if len(ridx[0]) == 0: 
                return 0
            ravg_distance = np.mean(raw_estimation[ridx[0]])
            ravg_angle = np.mod(np.mean(ridx[0] * self.beam_separation), 2*np.pi)

            d = np.exp(-(avg_distance - ravg_distance)**2/measurement_noise[0]**2)
            a = np.exp(-(avg_angle - ravg_angle) **2/measurement_noise[1]**2)
            likelihood *= d * a
            if likelihood == 0:
                return 0.0
        return likelihood


    def _best_match(self, raw_estimation, sensor_distances, measurement_noise, state):
        """
        Finds the best match between the sensor readings and the raw estimation and returs the likelihood
        """
        # we search in beams between -N/2 and N/2 to find the best match
        N = 360 
        re = self.range + self.resolution - raw_estimation
        nonzero_mask = re > 0
        n_active_beams = np.count_nonzero(nonzero_mask)
        nonzero_shifted_mask = (np.where(re > 0)[0].reshape(1,-1) + np.arange(-N//2, N//2).reshape(-1,1)) % self.n_beams
        similarity = (re[nonzero_mask] - (sensor_distances[nonzero_shifted_mask.reshape(-1)].reshape((nonzero_shifted_mask.shape[0],-1))))**2
        
        peaks_at = np.argmin(similarity, axis=0)
        p_z_given_x_distance = np.exp(-similarity[peaks_at,np.arange(n_active_beams)]/measurement_noise[0]**2)
        p_z_given_x_angle = np.exp(-np.abs(N//2 - peaks_at) * self.beam_separation/measurement_noise[1]**2)
        likelihood = p_z_given_x_distance * p_z_given_x_angle
        return likelihood.prod()

    def calibrate(self):
        pass


@staticmethod
def model(grid, robot_pose, sensor_pose, sensor_params):
    """
    """
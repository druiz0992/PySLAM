from abc import abstractmethod
import numpy as np
import sys


class Particles:
    def __init__(self, number_of_particles, state_dim):
        """
        Initialize the particles.

        :param number_of_particles: Number of particles
        :param state_dim: State dimension
        """
        self.number_of_particles = number_of_particles
        self.weights = np.ones(number_of_particles) / number_of_particles
        self.states = np.zeros((number_of_particles, state_dim))

    def get_state_dimensions(self):
        """
        Return state dimensions.

        :return: State dimensions
        """
        return self.states.shape[1]

    def init_uniform(self, limits):
        """
        Initialize the particles uniformly over the world. The limits are given as a list with maximum and minimum values

        :param limits: List with maximum and minimum values state dimensions: [xmin, xmax, ymin, ymax, zmin, zmax,...]

        NOTE: The number of elements in the limits vector must be equal to 2*state_dimension.
        """
        if len(limits)//2 != self.get_state_dimensions():
            print("Limits vector has incorrect length")
            sys.exit(1)

        self.weights = np.ones(self.number_of_particles) / self.number_of_particles
        for i in range(len(limits) // 2):
            self.states[:, i] = np.random.uniform(limits[2*i], limits[2*i+1])

    def init_gaussian(self, mean, std):
        """
        Initialize the particles using a Gaussian distribution. The mean and standard deviation are given as lists.

        :param mean: Mean of the Gaussian distribution used for initializing the particle states
        :param std: Standard deviations (one for each dimension)

        NOTE: The number of elements in the mean and std vectors must be equal to the state dimension.
        """
        state_dimensions = self.get_state_dimensions()
        if len(mean) != state_dimensions or len(std) != state_dimensions:
            print("Std/Mean vectors have incorrect length")
            sys.exit(1)

        self.weights = np.ones(self.number_of_particles) / self.number_of_particles
        for i in range(self.get_state_dimensions()):
            self.states[:, i] = np.random.normal(mean[i], std[i], self.number_of_particles)

    def sum_weights(self):
        """
        Compute the sum of all particle weights.

        :return: Sum of all particle weights
        """
        return np.sum(self.weights)
    
    def sum_weights_squared(self):
        """
        Compute the sum of all particle weights squared.

        :return: Sum of all particle weights squared
        """
        return np.sum(self.weights * self.weights)
    
    def set(self, particles):
        """
        Set particles.

        :param particles: Initial particle set: [[weight_1, [x1, y1, theta1]], ..., [weight_n, [xn, yn, thetan]]]
        """
        self.weights = np.array([p[0] for p in particles])
        self.states = np.array([p[1] for p in particles])
    
    def update(self, new_states, new_weights):
        """
        Update the particles with new states and weights.

        :param new_states: New states
        :param new_weights: New weights

        NOTE: The number of elements in the new_states and new_weights vectors must be equal to the number of particles.
        """
        if len(new_states) != self.number_of_particles or len(new_weights) != self.number_of_particles:
            print("New states/weights vectors have incorrect length")
            sys.exit(1)

        self.states = new_states
        self.weights = new_weights

    def get(self):
        return (self.states, self.weights)

    def get_weights(self):
        return self.weights 
    
    def get_states(self):
        return self.states
    
    def get_number_particles(self):
        return self.number_of_particles
    
class ParticleFilter:
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * Abstract class
    """

    def __init__(self, number_of_particles, limits, process_noise, measurement_noise):
        """
        Initialize the abstract particle filter.

        :param number_of_particles: Number of particles
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax]
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular]
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle]
        """

        if number_of_particles < 1:
            print("Warning: initializing particle filter with number of particles < 1: {}".format(number_of_particles))

        # Initialize filter settings
        self.n_particles = number_of_particles

        # State related settings
        self.state_dimension = 3  # x, y, theta
        self.x_min = limits[0]
        self.x_max = limits[1]
        self.y_min = limits[0]
        self.y_max = limits[1]

        # Set noise
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.particles = Particles(self.n_particles, self.state_dimension)

    def get_particle_states(self):
        """
        Return particle states.

        :return: Particle states.
        """
        return self.particles.get_states()
    
    def get_particle_weights(self):
        """
        Return particle weights.

        :return: Particle weights.
        """
        return self.particles.get_weights()
    
    def get_particles(self):
        """
        Return particles.

        :return: Particles.
        """
        return self.particles.get()

    def initialize_particles_uniform(self):
        """
        Initialize the particles uniformly over the world assuming a 3D state (x, y, heading). No arguments are required
        and function always succeeds hence no return value.
        """
        # Initialize particles with uniform weight distribution
        self.particles.init_uniform([self.x_min, self.x_max, self.y_min, self.y_max, 0, 2 * np.pi])

    def initialize_particles_gaussian(self, mean_vector, standard_deviation_vector):
        """
        Initialize particle filter using a Gaussian distribution with dimension three: x, y, heading. Only standard
        deviations can be provided hence the covariances are all assumed zero.

        :param mean_vector: Mean of the Gaussian distribution used for initializing the particle states
        :param standard_deviation_vector: Standard deviations (one for each dimension)
        :return: Boolean indicating success
        """

        # Check input dimensions
        if len(mean_vector) != self.state_dimension or len(standard_deviation_vector) != self.state_dimension:
            print("Means and state deviation vectors have incorrect length in initialize_particles_gaussian()")
            return False

        # Initialize particles with uniform weight distribution
        self.particles.init_gaussian(mean_vector, standard_deviation_vector)

    def validate_state(self, state):
        """
        Validate the state. State values outide allowed ranges will be corrected for assuming a 'cyclic world'.

        :param state: Input particle state.
        :return: Validated particle state.
        """

        # Make sure state does not exceed allowed limits (cyclic world)
        state[0] = min(max(self.x_min, state[0]), self.x_max)
        state[1] = min(max(self.x_min, state[1]), self.y_max)

        # Angle must be [-pi, pi]
        while state[2] > np.pi:
            state[2] -= 2 * np.pi
        while state[2] < -np.pi:
            state[2] += 2 * np.pi

        return state

    def set_particles(self, particles):
        """
        Initialize the particle filter using the given set of particles.

        :param particles: Initial particle set: [[weight_1, [x1, y1, theta1]], ..., [weight_n, [xn, yn, thetan]]]
        """

        # Assumption: particle have correct format, set particles
        self.particles = Particles(len(particles), 3)
        self.particles.set(particles)
        self.n_particles = len(self.n_particles)

    def get_average_state(self):
        """
        Compute average state according to all weighted particles

        :return: Average x-position, y-position and orientation
        """

        # Compute sum of all weights
        sum_weights = self.particles.sum_weights()

        # Compute weighted average
        avg_x = 0.0
        avg_y = 0.0
        avg_theta = 0.0
        avg_x = np.sum(self.particles.weights / sum_weights * self.particles.states[:, 0])
        avg_y = np.sum(self.particles.weights / sum_weights * self.particles.states[:, 1])
        avg_theta = np.sum(self.particles.weights / sum_weights * self.particles.states[:, 2])

        return [avg_x, avg_y, avg_theta]

    def get_max_weight(self):
        """
        Find maximum weight in particle filter.

        :return: Maximum particle weight
        """
        return np.max(self.particles.weights)


    def print_particles(self):
        """
        Print all particles: index, state and weight.
        """

        print("Particles:")
        for i in range(self.n_particles):
            print(" ({}): {} with w: {}".format(i+1, self.particles.states[i], self.particles.weights[i]))

    def normalize_weights(self):
        """
        Normalize all particle weights.
        """

        # Compute sum weighted samples
        sum_weights = self.particles.sum_weights()

        # Check if weights are non-zero
        if sum_weights < 1e-15:
            print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(sum_weights))

            # Set uniform weights
            self.particles.weights = np.ones(self.n_particles) / self.n_particles
            return

        # Return normalized weights
        self.particles.weights /= sum_weights


    def update_particles(self, weights, states):
        """
        Update the particle filter with new weights and states.

        :param weights: New weights
        :param states: New states
        """

        self.particles.update(states, weights)

    def propagate_samples(self, samples, forward_motion, angular_motion):
        """
        Propagate all samples with a simple motion model that assumes the robot rotates angular_motion rad and then moves
        forward_motion meters in the direction of its heading. Return the propagated samples (leave input unchanged).

        :param samples: List of samples (unweighted particles) that must be propagated
        :param forward_motion: Forward motion in meters
        :param angular_motion: Angular motion in radians
        :return: propagated samples
        """
        samples[:,2] += np.random.normal(angular_motion, self.process_noise[1], self.n_particles)
        forward_displacement = np.random.normal(forward_motion, self.process_noise[0], self.n_particles)
        samples[:,0] += forward_displacement * np.cos(samples[:,2])
        samples[:,1] += forward_displacement * np.sin(samples[:,2])
        #return self.validate_state(propagated_sample)
        return samples


    def compute_likelihood(self, sample, measurement, landmarks):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample state and landmarks.

        :param sample: Sample (unweighted particle) that must be propagated
        :param measurement: List with measurements, for each landmark [distance_to_landmark, angle_wrt_landmark], units
        are meters and radians
        :param landmarks: Positions (absolute) landmarks (in meters)
        :return Likelihood
        """

        # Initialize measurement likelihood
        likelihood_sample = np.ones(self.n_particles)

        skip_angle = np.sum(measurement[:,1]**2) < 1e-15
        p_z_given_x_angle = 1.0

        # Loop over all landmarks for current particle
        for i, lm in enumerate(landmarks):

            # Compute expected measurement assuming the current particle state
            dx = sample[:,0] - lm[0]
            dy = sample[:,1] - lm[1]
            expected_distance = np.sqrt(dx*dx + dy*dy)

            if not skip_angle:
                expected_angle = np.arctan2(dy, dx)
                # Map difference true and expected angle measurement to probability
                p_z_given_x_angle = \
                    np.exp(-((expected_angle-measurement[i][1])**2)/
                           (2 * self.measurement_noise[1]**2))

            # Map difference true and expected distance measurement to probability
            p_z_given_x_distance = \
                np.exp(-((expected_distance-measurement[i][0])**2) /
                       (2 * self.measurement_noise[0]**2))

    

            # Incorporate likelihoods current landmark
            likelihood_sample *= p_z_given_x_distance * p_z_given_x_angle

        # Return importance weight based on all landmarks
        return likelihood_sample

    @abstractmethod
    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement. Abstract method that must be implemented in derived
        class.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        pass

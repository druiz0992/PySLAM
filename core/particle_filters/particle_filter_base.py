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

    def init_from_distribution(self, old_states, valid_particles,  world_grid):
        """
        Initialize the particles using the given initialization function.

        :param init_function: Initialization function
        :param number_of_particles: Number of particles
        """
        vp_cntr = valid_particles

        # Need to generate particles until we have the correct number of valid particles, taking into account
        # the world grid
        collision_particles_mask = world_grid.is_occupied(old_states[:,:2], world_coordinates=True)
        vp = np.sum(collision_particles_mask == False)
        elements_to_copy = min(vp,self.number_of_particles-vp_cntr)
        tmp = old_states[collision_particles_mask == False]
        self.states[vp_cntr:vp_cntr+elements_to_copy] = tmp[:elements_to_copy, :]
        vp_cntr += vp
        return vp_cntr



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
    
    def set_states(self, states):
        self.states = states
    
    def set_weights(self, weights):
        self.weights = weights
    
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
        self.consecutive_weight_normalization = 0

        self.world = None
        self.sensors = []

    def install_sensors(self, sensor_list):
        """
        Install sensors.

        :param sensor_list: List with sensors
        """
        self.sensors = sensor_list

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

    def initialize_landmarks(self, landmarks):
        self.landmarks = landmarks

    def initialize_particles_uniform(self, world):
        """
        Initialize the particles uniformly over the world assuming a 3D state (x, y, heading). No arguments are required
        and function always succeeds hence no return value.
        """
        self.world = world
        # Initialize particles with uniform weight distribution
        limits = [self.x_min, self.x_max, self.y_min, self.y_max, 0, 2 * np.pi]
        s = np.zeros((self.n_particles, self.state_dimension))
        valid_particles = 0
        while valid_particles < self.n_particles:
            s[:,0] = np.random.uniform(limits[0], limits[1], self.n_particles)
            s[:,1] = np.random.uniform(limits[2], limits[3], self.n_particles)
            s[:,2] = np.random.uniform(limits[4], limits[5], self.n_particles)
            valid_particles = self.particles.init_from_distribution(s, valid_particles, self.world.grid())


    def initialize_particles_gaussian(self, mean_vector, standard_deviation_vector, world):
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

        self.world = world
        valid_particles = 0
        while valid_particles < self.n_particles:
            s = np.random.normal(mean_vector, standard_deviation_vector, (self.n_particles, self.state_dimension) )
            valid_particles = self.particles.init_from_distribution(s, valid_particles, self.world.grid())

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
        if sum_weights < 1e-15:
            return [0.0, 0.0, 0.0]
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

            self.consecutive_weight_normalization += 1
            if self.consecutive_weight_normalization > 50:
                print("Too many consecutive weight normalizations")
                self.initialize_particles_uniform(self.world)
                self.consecutive_weight_normalization = 0
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

    def propagate_samples(self, states, forward_motion, angular_motion):
        """
        Propagate all samples with a simple motion model that assumes the robot rotates angular_motion rad and then moves
        forward_motion meters in the direction of its heading. Return the propagated samples (leave input unchanged).

        :param samples: List of samples (unweighted particles) that must be propagated
        :param forward_motion: Forward motion in meters
        :param angular_motion: Angular motion in radians
        :return: propagated samples
        """
        valid_particles = 0
        while valid_particles < self.n_particles:
            states[:,2] += np.random.normal(angular_motion, self.process_noise[1], self.n_particles)
            forward_displacement = np.random.normal(forward_motion, self.process_noise[0], self.n_particles)
            states[:,0] += forward_displacement * np.cos(states[:,2])
            states[:,1] += forward_displacement * np.sin(states[:,2])
            valid_particles = self.particles.init_from_distribution(states, valid_particles, self.world.grid())
        # normalize angles to 0 - 2pi
        states[:,2] = np.mod(states[:,2], 2 * np.pi)   
        return self.particles.get_states()


    def compute_likelihood(self, sample, measurement):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample state and landmarks.

        :param sample: Sample (unweighted particle) that must be propagated
        :param measurement: List with measurements, for each landmark [distance_to_landmark, angle_wrt_landmark], units
        are meters and radians
        :param landmarks: Positions (absolute) landmarks (in meters)
        :return Likelihood
        """

        likelihood_sample = []
        for sensor_idx, sensor in enumerate(self.sensors):
            likelihood_sample.append(sensor.compute_likelihood(sample, measurement[sensor_idx], self.measurement_noise))
        
        return np.asarray(likelihood_sample).prod(axis=0)


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

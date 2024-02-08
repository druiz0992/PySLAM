from .particle_filter_base import ParticleFilter
from core.resampling.resampler import Resampler
import numpy as np


class ParticleFilterSIR(ParticleFilter):
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * propagation and measurement models are largely hardcoded (except for standard deviations.
    """

    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise,
                 resampling_algorithm,
                 **_):
        """
        Initialize the SIR particle filter.

        :param number_of_particles: Number of particles.
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax].
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular].
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle].
        :param resampling_algorithm: Algorithm that must be used for core.
        """
        # Initialize particle filter base class
        super().__init__(number_of_particles, limits, process_noise, measurement_noise)

        # Set SIR specific properties
        self.resampling_algorithm = resampling_algorithm
        self.resampler = Resampler()

    def needs_resampling(self):
        """
        Method that determines whether not a core step is needed for the current particle filter state estimate.
        The sampling importance core (SIR) scheme resamples every time step hence always return true.

        :return: Boolean indicating whether or not core is needed.
        """
        return True

    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement and resample if needed.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        # Loop over all particles
        particle_states = np.copy(self.get_particle_states())
        particle_weights = np.copy(self.get_particle_weights())

        propagated_states = self.propagate_samples(particle_states, robot_forward_motion, robot_angular_motion)
        w = particle_weights * self.compute_likelihood(propagated_states, measurements, landmarks)

        self.update_particles(w, propagated_states)

        # Update particles
        self.normalize_weights()

        # Resample if needed
        if self.needs_resampling():
            new_states, new_weights = self.resampler.resample(self.particles, self.n_particles, self.resampling_algorithm)
            self.particles.update(new_states, new_weights)

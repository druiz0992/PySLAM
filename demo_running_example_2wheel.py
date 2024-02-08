import numpy as np

# Simulation + plotting requires a robot, visualizer and world
from simulator.robots import TwoWheeledRobot
from simulator.world import World, WorldObject
from simulator import sample_motion_plan

# Supported resampling methods (resampling algorithm enum for SIR and SIR-derived particle filters)
from core.resampling import ResamplingAlgorithms

# Particle filters
from core.particle_filters import ParticleFilterNEPR,ParticleFilterSIR

# For showing plots (plt.show())
import matplotlib.pyplot as plt

if __name__ == '__main__':

    ## 
    # General settings
    ##
    sampling_time = 0.05

    ##
    # World settings
    ##
    world_size_x = 10.0
    world_size_y = 10.0
    n_landmarks = 4 

    ##
    # True robot properties (simulator settings)
    ##
    starting_robot_pose = [world_size_x * 0.5, world_size_y * 0.5, np.pi / 2.0]

    # True simulated robot motion is set point plus additive zero mean Gaussian noise with these standard deviation
    true_robot_rpm_std = 0.1
    true_wheel_diameter_std = 0.1
    true_wheel_distance_std = 0.1
    true_wheel_disalignment_std = 0.1

    # Robot measurements are corrupted by measurement noise
    true_robot_meas_noise_distance_std = 0.2
    true_robot_meas_noise_angle_std = 0.2

    # True robot properties
    true_wheel_distance_cm = 5.0
    true_wheel_diameter_cm = 2.0
    robot_size = 5.0
    skip_angle_measurement = False

    robot_params = {
        'wheel_distance': true_wheel_distance_cm,
        'wheel_diameter': true_wheel_diameter_cm,
        'skip_angle_measurement': skip_angle_measurement
    }

    robot_noise_params = {
        'std_rpm': true_robot_rpm_std,
        'std_wheel_diameter': true_wheel_diameter_std,
        'std_wheel_distance': true_wheel_distance_std,
        'std_disalignment': true_wheel_disalignment_std,
        'std_meas_distance': true_robot_meas_noise_distance_std,
        'std_meas_angle': true_robot_meas_noise_angle_std
    }

    # Robot Motion plan # left rpm, right rpm, duration seconds
    robot_motion_plan = [[100, 100, 2], [100, 50, 4], [100, 100, 2], [50, 100, 4],
                         [100, 100, 2], [100, 50, 4], [100, 100, 2], [50, 100, 4],
                         [100, 100, 2], [100, 50, 4], [100, 100, 2], [50, 100, 4]]
    true_motion_plan = sample_motion_plan(robot_motion_plan, sampling_time)


    # Initialize simulated robot
    robot = TwoWheeledRobot(starting_robot_pose,
                  robot_params = robot_params,
                  robot_noise_params = robot_noise_params)

    ##
    # Particle filter settings
    ##

    number_of_particles = 1000
    pf_state_limits = [0, world_size_x, 0, world_size_y]
    number_of_effective_particles_threshold = number_of_particles / 4.0

    # Process model noise (zero mean additive Gaussian noise)
    motion_model_forward_std = 0.1
    motion_model_turn_std = 0.20
    process_noise = [motion_model_forward_std, motion_model_turn_std]

    # Measurement noise (zero mean additive Gaussian noise)
    meas_model_distance_std = 0.4
    meas_model_angle_std = 0.3
    measurement_noise = [meas_model_distance_std, meas_model_angle_std]

    # Set resampling algorithm used
    algorithm = ResamplingAlgorithms.STRATIFIED

    # Initialize SIR particle filter: resample every time step
    particle_filter = ParticleFilterNEPR(
        number_of_particles=number_of_particles,
        limits=pf_state_limits,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        resampling_algorithm=algorithm,
        number_of_effective_particles_threshold=number_of_effective_particles_threshold)


    ##
    # Set simulated world and visualization properties
    ##
    show_particles = True
    visualizer_opts = {
        'draw_particles': show_particles,
        'particle_color': 'g',
        'robot_radius': 0.2,
        'landmark_size': 7,
        'robot_arrow_length': 0.5,
    }
    world_opts = {
        'world_size': [world_size_x, world_size_y],
        'n_landmarks': n_landmarks,
        'world_type': 'maze', #'reference_landmarks'
    }

    world = World(world_opts, visualizer_opts)
    reference_landmarks = world.landmarks.get_landmarks()[4:]
    reference_landmarks_coordinates = [landmark.get_center() for landmark in reference_landmarks]

    # check that the robot is not inside a landmark. If it is, select a new starting pose and check again
    while any([landmark.contains_point(robot.get_pose()[:2]) for landmark in reference_landmarks]):
        robot.set_initial_pose([np.random.uniform(1, world_size_x-2), np.random.uniform(1, world_size_y-2), np.random.uniform(0, 2 * np.pi)])

    particle_filter.initialize_particles_gaussian(robot.get_pose(), [meas_model_distance_std, meas_model_distance_std, meas_model_angle_std])

    ##
    # Start simulation
    ##
    timestamp = 0
    for motion in true_motion_plan:
        # Simulate robot motion (required motion will not exactly be achieved)
        desired_Lrpm = motion[0]
        desired_Rrpm = motion[1]
        robot.move([desired_Lrpm, desired_Rrpm], world, timestamp)

        raw_movement = robot.get_raw_incremental_movement()

        # Simulate measurement
        measurements = robot.measure(reference_landmarks_coordinates)

        # Update SIR particle filter
        particle_filter.update(robot_forward_motion=raw_movement[0],
                                   robot_angular_motion=raw_movement[1],
                                   measurements=measurements,
                                   landmarks=reference_landmarks_coordinates)
        
        estimated_pose = particle_filter.get_average_state()
        robot.set_estimated_pose(estimated_pose)

        # Visualization
        world.draw_world(robot.get_pose(), particle_filter.particles, robot.get_trajectory())
        timestamp += sampling_time
        plt.pause(sampling_time)

    error = robot.measure_error()
    print("Error: ", error)


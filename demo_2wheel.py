import numpy as np
from time import sleep

# Simulation + plotting requires a robot, visualizer and world
from simulator.robots import TwoWheeledRobot
from simulator.world import World
from simulator import sample_motion_plan
from simulator.sensors import SensorType

# Supported resampling methods (resampling algorithm enum for SIR and SIR-derived particle filters)
from core.resampling import ResamplingAlgorithms

# Particle filters
from core.particle_filters import ParticleFilterNEPR

if __name__ == '__main__':

    ## 
    # General settings
    ##
    sampling_time = 0.05

    ##
    # World settings
    ##
    world_size_x = 100.0
    world_size_y = 100.0
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
        'skip_angle_measurement': skip_angle_measurement,
        'world_size': [world_size_x, world_size_y]
    }

    robot_noise_params = {
        'std_rpm': true_robot_rpm_std,
        'std_wheel_diameter': true_wheel_diameter_std,
        'std_wheel_distance': true_wheel_distance_std,
        'std_disalignment': true_wheel_disalignment_std,
        'std_meas_distance': true_robot_meas_noise_distance_std,
        'std_meas_angle': true_robot_meas_noise_angle_std
    }
    
    ## 
    # sensors - Gyro

    gyro_params = {
        'std_bias': 1.5, #rad/sqrt(h)
        'mean_bias': 0.5, #rad/h
        'std_noise': 1.5 #rad/sqrt(h)
    }

    sensor_params = {
        'gyro': gyro_params
    }

    # LIDAR
    lidar_params = {
        'n_beams': 360,
        'n_scans': 10,
        'fov': np.pi/180, # 1 degree
        'range': 200,
        'beam_offset': 0.0,
        'resolution': 0.1,
        'std_noise': 0.2,
        'z_hit': 1.0,
        'z_short': 0,
        'z_max': 0.0,
        'z_rand': 0.0,
        'lambda_short': 0.1,
        'skip_angle_measurement': False,
    }
    sensor_params = {
        SensorType.LIDAR: lidar_params
    }
    

    # Robot Motion plan # left rpm, right rpm, duration seconds
    robot_motion_plan = [[100, 100, 2], [100, 50, 4], [100, 100, 2], [50, 100, 4],
                         [100, 100, 2], [100, 50, 4], [100, 100, 2], [50, 100, 4],
                         [100, 100, 2], [100, 50, 4], [100, 100, 2], [50, 100, 4]]
    true_motion_plan = sample_motion_plan(robot_motion_plan, sampling_time)


    # Initialize simulated robot
    robot = TwoWheeledRobot(starting_robot_pose,
                  robot_params = robot_params,
                  robot_noise_params = robot_noise_params,
                  sensor_params = sensor_params)

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
    meas_model_distance_std = 2
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

    remove_walls_from_grid = True
    grid_scale = [10, 10]
    world_opts = {
        'world_size': [world_size_x, world_size_y],
        'n_landmarks': n_landmarks,
        'world_type': 'fixed_landmarks', #'maze', #'reference_landmarks'
        'grid_scale': grid_scale,
        'remove_walls_from_grid': remove_walls_from_grid
    }

    # Create world. After creating world, landmarks are added to it, so robot and particles must be initialized
    world = World(world_opts, visualizer_opts)

    # Initialize grid in robot. It is used to ensure robot doesnt go through walls
    robot.initialize(world, update_pose_if_collision=True)

    # Initialize particles randomly, but avoid placing them in occupied cells
    particle_filter.initialize_particles_uniform(world)
    #particle_filter.initialize_particles_gaussian(
     #   robot.get_pose(), 
      #  [meas_model_distance_std, meas_model_distance_std, meas_model_angle_std],
       # world)

    # install sensors in particle filter
    particle_filter.install_sensors(robot.sensors_as_list())

    ##
    # Start simulation
    ##
    timestamp = 0

    for desired_rpm in true_motion_plan:
        # update robot's motion according to motion plan
        raw_movement = robot.move(desired_rpm, timestamp)

        # Robot measures world state
        measurements = robot.measure(timestamp, add_noise=True)

        # Update SIR particle filter
        particle_filter.update(robot_forward_motion=raw_movement[0],
                                   robot_angular_motion=raw_movement[1],
                                   measurements=measurements,
                                   add_noise=True)
                                   #pose=robot.get_pose() + [0,0,0])
        
        robot.set_estimated_pose(particle_filter.get_average_state())

        # Visualization
        world.render(robot.get_pose(), particle_filter.particles, robot.get_trajectory())
        timestamp += sampling_time
        sleep(sampling_time)

    error = robot.measure_error()
    print("Error: ", error)


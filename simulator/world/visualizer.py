# Plotting will be done with matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numpy as np

class Visualizer:
    """
    Class for visualing the world, the true robot pose and the discrete distribution that estimates the robot pose by
    means of a set of (weighted) particles.
    """

    def __init__(self, world_dims, landmark_vertices, opts):
        """
        Initialize visualizer. By setting the flag to false the full 2D pose will be visualized. This makes
        visualization much slower hence is only recommended for a relatively low number of particles.

        :param draw_particles: Boolean that indicates whether particles must be drawn or not.
        """

        self.x_max = world_dims[0]
        self.y_max = world_dims[1]
        self.circle_radius_robot = opts['robot_radius']
        self.draw_particles = opts['draw_particles']
        self.landmark_size = opts['landmark_size']
        self.robot_arrow_length = opts['robot_arrow_length']
        self.particle_color = opts['particle_color']
        self.landmark_vertices = landmark_vertices

    def render(self, robot_pose, particles, trajectory=None):
        """
        Draw the simulated world with its landmarks, the robot 2D pose and the particles that represent the discrete
        probability distribution that estimates the robot pose.

        :param world: World object (includes dimensions and landmarks)
        :param robot: True robot 2D pose (x, y, heading)
        :param particles: Set of weighted particles (list of [weight, [x, y, heading]]-lists)
        :param hold_on: Boolean that indicates whether figure must be cleared or nog
        :param particle_color: Color used for particles (as matplotlib string)
        """

        # Draw world
        plt.figure(1, figsize=(self.x_max + 1, self.y_max + 1))
        plt.clf()
        ax = plt.gca()

        # Set limits axes
        plt.xlim([0, self.x_max])
        plt.ylim([0, self.y_max])

        # No ticks on axes
        plt.xticks([])
        plt.yticks([])

        # Add landmarks
        for vertices in self.landmark_vertices:
           ax.add_patch(Polygon(vertices, closed=True, facecolor='blue'))  

        # Add particles
        if self.draw_particles: 
             # Convert to numpy array for efficiency reasons (drop weights)
             states = particles.get_states()
             plt.plot(states[:, 0], states[:, 1], self.particle_color+'.', linewidth=1, markersize=2)

        # Add robot pose
        self.add_pose2d(robot_pose, 1, 'r', self.circle_radius_robot)

        # Draw robot trajectory
        if trajectory is not None:
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'g-', linewidth=1)
            plt.plot(np.mod(trajectory[:, 3], self.x_max), np.mod(trajectory[:, 4], self.y_max), 'b-', linewidth=1)
            plt.scatter(np.mod(trajectory[:, 7], self.x_max), np.mod(trajectory[:, 8], self.y_max), color='red',s=1)

        # Show
        plt.pause(0.05)

    def add_pose2d(self, robot_pose, fig_num, color, radius):
        """
        Plot a 2D pose in given figure with given color and radius (circle with line indicating heading).

        :param x: X-position (center circle).
        :param y: Y-position (center circle).
        :param theta: Heading (line from center circle with this heading will be added).
        :param fig_num: Figure in which pose must be added.
        :param color: Color of the lines.
        :param radius: Radius of the circle.
        :return:
        """

        x = robot_pose[0]
        y = robot_pose[1]
        theta = robot_pose[2]

        # Select correct figure
        plt.figure(fig_num)

        # Draw circle at given position (higher 'zorder' value means draw later, hence on top of other lines)
        circle = plt.Circle((x, y), radius, facecolor=color, edgecolor=color, alpha=0.4, zorder=20)
        plt.gca().add_patch(circle)

        # Draw line indicating heading
        plt.plot([x, x + radius * np.cos(theta)],
                 [y, y + radius * np.sin(theta)], color)

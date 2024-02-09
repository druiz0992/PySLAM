import numpy as np
import cv2

class WorldObject:
    """
    A 2D object in the world.
    
    Attributes:
    - vertices: List of 2D vertices defining the object
    - bounding_box: Bounding box of the object


    """

    def __init__(self, vertices, rotation, traslation = [0,0]):
        """
        Initialize the world object with given vertices and rotation.

        :param vertices: List of 2D vertices defining the object
        :param rotation: Rotation of the object (rad)
        """
        rotation_matrix = cv2.getRotationMatrix2D((0, 0), np.degrees(rotation), 1)
        self.vertices = cv2.transform(np.array([vertices], dtype=np.float32), rotation_matrix)[0]
        self.vertices = self.vertices + traslation
        self.vertices.flags.writeable = False

        # Bounding box is a convex polygon. It is used to check for collision and containment
        self.bounding_box = np.array(self.vertices, dtype=np.int32)
        self.bounding_box.flags.writeable = False

    def bounding_box_as_array(self):
        """
        Return the bounding box of the object.

        :return: Bounding box of the object
        """
        return self.bounding_box

    def vertices_as_array(self):
        """
        Return the vertices of the object.

        :return: List of 2D vertices defining the object
        """
        return self.vertices
    
    def center(self):
        """
        Return the center of the object.

        :return: 2D point defining the center of the object
        """
        return np.mean(self.vertices, axis=0)
    
    def intersects(self, other):
        """
        Check if the object intersects with another object.

        :param other: Other object to check for collision
        :return: True if the objects collide, False otherwise
        """
        return cv2.intersectConvexConvex(self.bounding_box_as_array(), other.bounding_box_as_array())[0] > 1e-8
    
    def contains_point(self, point):
        """
        Check if a point is inside the object.

        :param point: 2D point to check
        :return: True if the point is inside the object, False otherwise
        """
        return cv2.pointPolygonTest(self.bounding_box_as_array(), point, False) >= 0
    
    
    @staticmethod
    def square(center, size, rotation):
        """
        Create a square object.

        :param center: 2D point defining the center of the square
        :param size: Size of the square
        :param rotation: Rotation of the square (rad)
        :return: WorldObject representing the square
        """
        half_size = size / 2
        vertices = np.array([
            [half_size, half_size],
            [-half_size, half_size],
            [-half_size, -half_size],
            [half_size, -half_size]
        ])
        return WorldObject(vertices, rotation, center)
    
    @staticmethod
    def circle(center, radius):
        """
        Create a circle object.

        :param center: 2D point defining the center of the circle
        :param radius: Radius of the circle
        :return: WorldObject representing the circle
        """
        return WorldObject(np.array([[radius * np.cos(theta), radius * np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 100)]) + center, 0)
    
    @staticmethod
    def rectangle(center, width, height, rotation):
        """
        Create a rectangle object.

        :param center: 2D point defining the center of the rectangle
        :param width: Width of the rectangle
        :param height: Height of the rectangle
        :param rotation: Rotation of the rectangle (rad)
        :return: WorldObject representing the rectangle
        """
        half_width = width / 2
        half_height = height / 2
        vertices = np.array([
            [half_width, half_height],
            [-half_width, half_height],
            [-half_width, -half_height],
            [half_width, -half_height]
        ])
        return WorldObject(vertices, rotation, center)
    
    @staticmethod
    def triangle(center, size, rotation):
        """
        Create a triangle object.

        :param center: 2D point defining the center of the triangle
        :param size: Size of the triangle
        :param rotation: Rotation of the triangle (rad)
        :return: WorldObject representing the triangle
        """
        vertices = np.array([
            [0, size],
            [size * np.sqrt(3) / 2, -size / 2],
            [-size * np.sqrt(3) / 2, -size / 2]
        ])
        return WorldObject(vertices, rotation, center)
    

    @staticmethod
    def polygon(vertices):
        """
        Create a polygon object.

        :param vertices: List of 2D vertices defining the polygon
        :return: WorldObject representing the polygon
        """
        return WorldObject(vertices, 0)
    
    @staticmethod
    def random_polygon(num_vertices):
        """
        Create a random polygon object.

        :param center: 2D point defining the center of the polygon
        :param size: Size of the polygon
        :param num_vertices: Number of vertices of the polygon
        :return: WorldObject representing the random polygon
        """
        return WorldObject(np.random.rand(num_vertices, 2), 0)


    




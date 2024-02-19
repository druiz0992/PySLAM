
from enum import Enum
from abc import ABC, abstractmethod

class SensorType(Enum):
    ACCELEROMETER = 1
    GYROSCOPE = 2
    LIDAR = 3
    ULTRASONIC = 4
    WHEEL_ODOMETRY = 5

class Sensor(ABC):
    """
    Sensor class

    Attributes:
    pose (tuple): position of the sensor on the robot
    type (SensorType): type of the sensor
    
    """
    def __init__(self, world):
        self.world = world

    @abstractmethod
    def scan(self):
        pass

    @abstractmethod
    def calibrate(self):
        pass

    @abstractmethod
    def compute_likelihood(self):
        pass

    @abstractmethod
    def sample_sensor_model(self):
        pass
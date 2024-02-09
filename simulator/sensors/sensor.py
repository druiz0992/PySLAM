
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
    position (tuple): position of the sensor on the robot
    type (SensorType): type of the sensor
    
    """
    def __init__(self, position, type):
        self.position = position
        self.type = type

    def position(self):
        return self.position
    
    def update_position(self, new_position):
        self.position = new_position

    @abstractmethod
    def measure(self):
        pass

    @abstractmethod
    def calibrate(self):
        pass
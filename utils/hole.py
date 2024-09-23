import pymunk

from utils.helper import *



class Hole:
    def __init__(self, space, position):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.shape = Circle(self.body, HOLE_RADIUS, (0, 0))
        self.shape.color = (0, 0, 0, 255)
        self.shape.sensor = True
        self.shape.collision_type = collision_types["hole"]
        # shape.friction = 1
        space.add(self.body, self.shape)

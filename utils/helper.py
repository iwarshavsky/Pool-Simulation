
import math

import numpy as np
import pymunk
from pymunk import Vec2d
import pygame

FPS = 200

POSITION = 1
ANGLE = 2
SCORE_PANEL_HEIGHT = 50
BOARD_HEIGHT = 600
WINDOW_WIDTH = 1200
BALL_RADIUS = 25
BG = (50, 50, 50)
WHITE = (255, 255, 255)
balls_dist_from_left = 300
y_spacing = 8
NO_VELOCITY = Vec2d(0, 0)
HOLE_RADIUS = 40
ADD_HOLES_RADIUS = 40
collision_types = {
    "cue_ball": 1,
    "ball": 2,
    "hole": 3,
}
TEMP_SHAPE_EVENT = pygame.USEREVENT + 1


class Circle(pymunk.Circle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = None



def to_polar(arr):
    x, y = arr
    return np.array([math.atan2(y, x), math.sqrt(x ** 2 + y ** 2)])


def to_cartesian(arr):
    theta, r = arr
    return np.array([r * math.cos(theta), r * math.sin(theta)])


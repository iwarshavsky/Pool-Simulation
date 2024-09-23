import pymunk



from utils.helper import BALL_RADIUS, collision_types
from utils.helper import Circle



class Ball:
    def __init__(self, space, pos, tag=None):
        self.mass = 3
        self.inertia = pymunk.moment_for_circle(self.mass, 0, BALL_RADIUS, (0, 0))
        self.body = pymunk.Body(self.mass, self.inertia)
        self.body.position = pos
        self.shape_outer = Circle(self.body, BALL_RADIUS, (0, 0))
        self.shape_outer.friction = 1
        self.shape_outer.elasticity = 0.8
        self.shape_inner = Circle(self.body, 5, (0, 0))
        self.pivot = pymunk.PivotJoint(space.static_body, self.body, (0, 0), (0, 0))
        self.pivot.max_bias = 0  # disable joint correction
        self.pivot.max_force = 80  # emulate linear friction
        self.shape_inner.collision_type = collision_types["ball"]  # collision_types["cue_ball"] if is_cue_ball else
        self.shape_inner.tag = tag

        self.shape_inner.color = (255, 255, 255, 255)

        self.body.prev_velocity = None
        self.body.p = self
        self.shape_outer.color = (0, 0, 0, 255)
        space.add(self.body, self.shape_outer, self.shape_inner, self.pivot)


class CueBall(Ball):
    def __init__(self, space, pos):
        super().__init__(space, pos, "CueBall")
        # self.shape_inner.collision_type = collision_types["cue_ball"]
        self.shape_outer.color = (255, 255, 255, 255)

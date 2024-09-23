import math
from copy import deepcopy
import pygame

from utils.helper import *
from utils.ball import Ball
from utils.ball import CueBall
from pymunk import pygame_util

from utils.hole import Hole


class Simulator:
    def __init__(self, visible):

        pygame.init()
        self.font = pygame.font.SysFont('Lato', 30)

        self.visible = visible
        if visible:
            self.screen = pygame.display.set_mode(Vec2d(WINDOW_WIDTH, BOARD_HEIGHT))
            pygame.display.set_caption("Pool")

            self.bg = pygame.image.load("utils/bg.png").convert_alpha()

        self.clock = pygame.time.Clock()
        self.FPS = FPS if visible else 0

    def clear_screen(self):
        if self.visible:
            # self.screen.fill((255, 255, 255))
            self.screen.blit(self.bg, (0, 0))

    def tick(self):
        if self.visible:
            pygame.display.flip()
        self.clock.tick(self.FPS)


    def wait_for_space(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return

    def ignore_events(self):
        for event in pygame.event.get():  # Ignore events during turn
            pass



class GameState:
    """
    A state of the game. Each state includes its own physics simulator, and ball, wall and hole objects.
    """

    def __init__(self, visible, ball_positions=None, cue_ball_position=None):

        self.temp_shape = None


        self.space = pymunk.Space()
        self.space.damping = 0.9
        self.space.gravity = (0.0, 0.0)
        self.sim = Simulator(visible=visible)
        add_walls(self.space)
        self.holes = add_holes(self.space)
        self.visible = visible
        if ball_positions and cue_ball_position:
            self.cue_ball = CueBall(self.space, cue_ball_position)
            self.balls = add_balls(self.space, ball_positions)
        else:
            self.balls = add_balls(self.space)
            self.cue_ball = CueBall(self.space, (WINDOW_WIDTH * 3 / 4, BOARD_HEIGHT / 2))
        self.balls.add(self.cue_ball)
        self.ball_bodies_to_remove = set()
        self.draw_options = None if not self.visible else pymunk.pygame_util.DrawOptions(self.sim.screen)

        ch_ball = self.space.add_collision_handler(collision_types["ball"], collision_types["hole"])
        ch_ball.data["ball_bodies_to_remove"] = self.ball_bodies_to_remove
        ch_ball.pre_solve = ball_hit_hole

        if self.visible:
            self._sim_step()

    @staticmethod
    def evaluate(state):
        """
        Determine if the game was won, lost or no result, given the state
        :param state: a game state.
        :return: True if won, False is lost, None in should continue
        """
        # count number of balls
        # Check if cue ball is there
        if state.cue_ball:
            if len(state.balls) == 1:
                # Won
                return True
            else:
                # Continue

                return None
        else:
            # Lost
            return False


    def copy(self, visible=False):

        """
        Creates a copy of the current state
        :return: the new state
        """


        new_state = GameState(visible=visible, ball_positions=self.get_ball_positions(),
                              cue_ball_position=self.get_cue_ball_position())
        return new_state

    def do_action(self, angle, speed, visible=False, show_target=False, display_text=""):

        """
        Creates a copy of the current state, fires the cue ball at the angle and speed.
        :param angle:
        :param speed:
        :param visible: Should the simulation be visible
        :param show_target: show what point we are aiming at
        :param display_text: What text to display while showing the target
        :return: resulting state
        """

        new_state = self.copy(visible)
        if show_target and visible:

            if new_state.temp_shape is None:
                new_state.temp_shape = (new_state.cue_ball.body.position,
                                        new_state.cue_ball.body.position + pymunk.Vec2d(speed / 3, 0).rotated(angle),
                                        display_text)
                pygame.time.set_timer(TEMP_SHAPE_EVENT, 2000)

        velocity = (speed * math.cos(angle), speed * math.sin(angle))

        new_state.cue_ball.body.velocity = velocity
        steps = 0

        counter = 100
        turn_over = False
        while not turn_over:
            if new_state.cue_ball is None:
                break
            if counter > 0:
                counter -= 1
            else:
                movement_detected = False
                for ball in new_state.balls:

                    # We round the velocity a bit so we don't get stuck on tiny numbers
                    v_rounded = ((round(ball.body.velocity.x, 2)), round(ball.body.velocity.y, 2))
                    prev_v_rounded = None
                    if ball.body.prev_velocity:
                        prev_v_rounded = ((round(ball.body.prev_velocity.x, 2)), round(ball.body.prev_velocity.y, 2))
                    if not (prev_v_rounded == NO_VELOCITY and v_rounded == NO_VELOCITY):

                        movement_detected = True
                    ball.body.prev_velocity = ball.body.velocity
                if not movement_detected:
                    return new_state

            new_state._sim_step()

            steps += 1

        return new_state

    def _sim_step(self):
        while self.ball_bodies_to_remove:
            ball_body = self.ball_bodies_to_remove.pop()
            if self.cue_ball:
                if ball_body is self.cue_ball.body:
                    self.cue_ball = None
            shapes = list(ball_body.shapes)
            constraints = list(ball_body.constraints)
            self.space.remove(*constraints, *shapes, ball_body)
            self.balls.remove(ball_body.p)
        if self.visible:

            self.sim.clear_screen()
            if self.temp_shape:
                # pygame.draw.circle(self.draw_options.surface, (255, 0, 0), shape[1], round(3), 20)
                pygame.draw.line(self.draw_options.surface, (255, 0, 0), self.temp_shape[0], self.temp_shape[1], 3)
                img = self.sim.font.render(self.temp_shape[2], True, (255, 255, 255))
                self.sim.screen.blit(img, (50, 520))

            for event in pygame.event.get():
                if event.type == TEMP_SHAPE_EVENT and self.temp_shape is not None:
                    self.temp_shape = None
                    pygame.time.set_timer(TEMP_SHAPE_EVENT, 0)
                # Ignore events during turn
                pass

            self.space.debug_draw(self.draw_options)
        self.space.step(1 / FPS)
        self.sim.tick()

    def get_num_remaining_balls(self):
        """
        :return: Number of balls excluding cue ball
        """
        return len(self.balls) - 1

    def get_ball_positions(self):
        """
        :return: Return list of ball positions
        """
        return [b.body.position for b in self.balls if b is not self.cue_ball]

    
    def get_ball_positions_sorted(self):
        """
        :return: Return list of ball positions
        """
        return [ball for ball, _ in sorted([(b.body.position, b.shape_inner.tag)for b in self.balls if b is not self.cue_ball], key=lambda x : x[1])]



    def get_cue_ball_position(self):
        """
        :return: returns the cue_ball position
        """
        if self.cue_ball:
            return self.cue_ball.body.position

    def get_hole_positions(self):
        """
        :return: Returns the position of the holes (targets)
        """
        return [h.body.position for h in self.holes]


def add_balls(space, positions=None):
    balls = set()
    if positions:
        for c, position in enumerate(positions):
            balls.add(Ball(space, position, tag=c))
    else:
        balls_start_pos = (balls_dist_from_left, BOARD_HEIGHT / 2 - (4 * BALL_RADIUS + 2 * y_spacing))
        offset = Vec2d(0, 0)
        c = 0
        for i in range(5, 0, -1):
            for j in range(i):
                balls.add(Ball(space, balls_start_pos + offset + Vec2d(0, BALL_RADIUS * 2 * j + y_spacing * j), tag=c))
                c += 1
            offset += (BALL_RADIUS * 2, BALL_RADIUS + y_spacing / 2)
    return balls


def add_walls(space):
    # return
    base_width_vertical_wall = (BOARD_HEIGHT - (2 * ADD_HOLES_RADIUS))
    base_width_horizontal_wall = (WINDOW_WIDTH - (4 * ADD_HOLES_RADIUS)) / 2
    top_gap = 0.1  # percentage
    height = 20
    vertices = lambda w: [(0, 0), (w, 0), (w * (1 - top_gap), height), (w * top_gap, height)]
    walls_config = [(base_width_horizontal_wall, (ADD_HOLES_RADIUS, 0), 0)]
    walls_config += [(base_width_horizontal_wall, (ADD_HOLES_RADIUS * 3 + base_width_horizontal_wall, 0), 0)]
    walls_config += [(base_width_vertical_wall, (WINDOW_WIDTH, ADD_HOLES_RADIUS), math.pi / 2)]
    walls_config += [
        (base_width_horizontal_wall, (ADD_HOLES_RADIUS + base_width_horizontal_wall, BOARD_HEIGHT), math.pi)]
    walls_config += [
        (base_width_horizontal_wall, (ADD_HOLES_RADIUS * 3 + base_width_horizontal_wall * 2, BOARD_HEIGHT), math.pi)]
    walls_config += [(base_width_vertical_wall, (0, BOARD_HEIGHT - ADD_HOLES_RADIUS), 1.5 * math.pi)]

    for config in walls_config:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = config[POSITION]
        body.angle = config[ANGLE]
        poly = pymunk.Poly(body, vertices(config[0]))
        poly.elasticity = 0.8
        poly.color = (88, 57, 39, 255)
        space.add(body, poly)


def add_holes(space):
    coords_lst = [(0, 0), (WINDOW_WIDTH / 2, 0), (WINDOW_WIDTH, 0), (0, BOARD_HEIGHT),
                  (WINDOW_WIDTH / 2, BOARD_HEIGHT), (WINDOW_WIDTH, BOARD_HEIGHT)]
    holes = set()
    for position in coords_lst:
        holes.add(Hole(space, position))
    return holes


def ball_hit_hole(arbiter, space, data):
    # find ball
    for b in arbiter.shapes:
        if b.tag is not None:
            data["ball_bodies_to_remove"].add(b.body)
    return True

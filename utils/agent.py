import math
import random
import sys

import matplotlib
import numpy as np
import pygame
import pymunk
from pymunk import pygame_util
from pymunk import Vec2d


import utils.helper

from utils.helper import to_polar, to_cartesian


class Agent:
    """
    Abstract class for an agent
    """

    def getAction(self, state):
        """
        :param state: state to consider
        :return: Returns a chosen angle and direction given the input state
        """
        raise NotImplementedError()


class RandomAgent(Agent):

    """Completely random agent"""

    def getAction(self, state):
        angle = random.uniform(0.0, math.pi * 2)
        speed = random.uniform(100, 2000)
        return angle, speed


class ManualAgent(Agent):
    """Agent which accepts manual actions to perform"""

    def __init__(self, actions):
        self.counter = 0
        self.actions = actions

    def getAction(self, state):
        action = self.actions[self.counter]
        self.counter += 1
        return action


    def getAction(self, state):
        assert not state.visible is False
        angle = 0
        distance = 0
        while True:
            if state.cue_ball:
                mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), state.sim.screen)
                angle = (mouse_position - state.cue_ball.body.position).angle
                distance = (mouse_position - state.cue_ball.body.position).length * 4
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:

                    return angle, distance


class LocalSearchAgent(Agent):

    def __init__(self, debug=False, draw=False):
        """
        :param debug: Shows visual aid
        """
        self.func = None
        self.debug = debug
        self.draw = draw

    def guess(self, state):
        """
        Guess the next move out of the center of mass of the balls or a random position on the board.
        :param state:
        :return:
        """

        def center_of_mass():
            ball_positions = list(state.get_ball_positions())
            s = Vec2d(0, 0)
            for pos in ball_positions:
                s += pos

            init_target = s / len(ball_positions)
            cue_ball_position = state.get_cue_ball_position()
            angle = (init_target - cue_ball_position).angle
            distance = (init_target - cue_ball_position).length
            x0 = np.array([angle, distance])
            return x0

        r = np.random.uniform(0, 1)
        if r < 0.25:  # take a risk
            return center_of_mass()
        angle = random.uniform(0.0, math.pi * 2)
        speed = random.uniform(100, 2000)
        return np.array([angle, speed])

    def getAction(self, state):
        """
        Method which overrides the class method.
        :param state: current state
        :return: the choice of action
        """
        Tmin, Tmax, = 1, 100
        E_thresh = 0

        # Create an instance of the loss function belonging to the current state.
        # Each state has its unique function.
        self.func = Func(state)

        T = np.inf
        good_result = False
        iter_count = 0
        x = Ex = x_movement = x_num_balls_removed = None
        start_score = self.func.compute(np.array([0, 0]))
        x_best, Ex_best = None, np.inf
        best_num_balls_removed = 0
        restart = True
        neighbor_chosen_count = 0
        while not good_result:

            if restart:
                neighbor_chosen_count = 0
                best_num_balls_removed, num_balls_removed = 0, 0

                # Ensure our starting position is legal
                is_cue_ball = False
                while not is_cue_ball:
                    x = self.guess(state)
                    Ex, x_movement, num_balls_removed, is_cue_ball = self.func.compute(x, return_movement=True)

                if self.debug: print("First:", to_cartesian(x), Ex)
                if self.draw:
                    self.func.compute(x, visible=True, show_target=True, display_text="Random choice")
                x_best, Ex_best = x.copy(), Ex
                if num_balls_removed > best_num_balls_removed: best_num_balls_removed = num_balls_removed

            T = Tmax

            while T > Tmin and Ex > E_thresh:

                dE = np.inf

                neighbor_chosen = False
                if x_movement:  # any ball has moved

                    x_new, gradient = self.neighbor(x)

                    Ex_new, x_new_movement, num_balls_removed, is_cue_ball = self.func.compute(x_new,
                                                                                               return_movement=True)

                    dE = Ex_new - Ex

                    if dE < 0 and is_cue_ball:  # We prefer candidates with lower cost

                        x, x_movement, Ex, x_cue_ball = x_new.copy(), x_new_movement, Ex_new, is_cue_ball
                        if num_balls_removed > best_num_balls_removed: best_num_balls_removed = num_balls_removed
                        if self.debug: print("\tGood Neighbor", to_cartesian(x), Ex)
                        if self.draw: self.func.compute(x, visible=True, show_target=True, display_text="Neighbor")
                        neighbor_chosen = True
                    else:
                        # if self.debug: print("Bad Neighbor: ", to_cartesian(x_new), Ex_new)
                        if self.draw: self.func.compute(x_new, visible=True, show_target=True,
                                                        display_text="Bad Neighbor - disregarding")
                        dE = np.inf

                if dE >= -3:  # Even if the was neighbor slightly improved us, try to take a random guess.
                    x_new_random = self.guess(state)

                    Ex_new_random, x_new_random_movement, num_balls_removed, is_cue_ball = self.func.compute(
                        x_new_random, return_movement=True)

                    if is_cue_ball and (Ex_new_random < Ex or np.random.uniform(0, 1) < (T / Tmax)):

                        x, x_movement, Ex, x_cue_ball = x_new_random.copy(), x_new_random_movement, Ex_new_random, is_cue_ball
                        if num_balls_removed > best_num_balls_removed: best_num_balls_removed = num_balls_removed

                        if self.debug: print("\tTook risk", to_cartesian(x), Ex)
                        if self.draw: self.func.compute(x, visible=True, show_target=True,
                                                        display_text="Random choice between center of mass and random point")
                        neighbor_chosen = False
                T /= 1.75 * (best_num_balls_removed + 1)
                neighbor_chosen_count += neighbor_chosen
                if Ex_best > Ex:
                    if self.debug: print("*Updated best so far*")
                    x_best = x.copy()
                    Ex_best = Ex

            if state.do_action(x_best[0], x_best[1]).cue_ball is not None and Ex_best < start_score:
                # If legal move and it improved us.
                good_result = True  # Breaks the loop
                if self.debug: print("USING ", to_cartesian(x_best), Ex_best)
            else:  # If "bad" move, run this again once, if failed again choose a legal move randomly.
                if self.debug: print("Running again")
                if iter_count > 0:
                    # Just find a random legal move and do it.
                    while True:
                        x = self.guess(state)
                        if state.do_action(x[0], x[1]).cue_ball is not None:
                            return x[0], x[1]
                restart = True
                iter_count += 1

        if self.draw: self.func.compute(x_best, visible=True, show_target=True, display_text="Final choice")

        return x_best[0], x_best[1]

    def neighbor(self, x):
        """
        Produce a neighbor by calculating an approximation to the gradient of x and making a descent step
        :param x: the point whose neighbor we want to find.
        :return: the neighbor (in polar coordinates)
        """
        # Attempt to use approximation of gradient to predict neighbor

        g = self.func.compute_grad(x)
        norm = np.linalg.norm(g)
        eta = np.exp(-norm * 0.1) * 150
        a = (to_polar(to_cartesian(x) - eta * g), g)
        return a


class Func:
    """
    A class for the loss function of LocalSearch agent from [0,2pi]xR -> R, to be optimized.
    """

    def __init__(self, state):
        self.func = self.func_to_optimize(state)

    def func_to_optimize(self, state):
        def f(x, return_movement=False, visible=False, show_target=False, display_text=""):
            """
            Returns the sum of distances of the balls to their nearest hole, after doing an action of angle and distance
            :param x: The action to perform: distance and angle
            :param return_movement: should output more information received from running the simulation with this action
            :param visible: should the evaluation be done on a visible board
            :param show_target: Should a line be drawn to indicate direction and speed of hit.
            :param display_text: What text to display while showing the target
            :return:
            """
            angle = x[0]
            distance = x[1]
            new_state = state.do_action(angle, distance, visible=visible, show_target=show_target,
                                        display_text=display_text)
            ball_positions = new_state.get_ball_positions()
            hole_positions = new_state.get_hole_positions()
            movement_detected = len(set(state.get_ball_positions()).difference(set(ball_positions))) > 0
            num_balls_removed = len(state.get_ball_positions()) - len(ball_positions)
            loss = sum(
                [min([np.log(np.sqrt((ball_pos.x - hole_pos.x) ** 2 + (ball_pos.y - hole_pos.y) ** 2))
                      for hole_pos in hole_positions])
                 for ball_pos in ball_positions]) * 2 * len(ball_positions)

            if return_movement:
                return loss, movement_detected, num_balls_removed, new_state.cue_ball is not None
            return loss

        return f

    def compute(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def compute_grad(self, *args):
        """
        Compute an approximation of the gradient using partial derivatives with a finite delta
        :param args: angle and distance
        :return: the gradient
        """
        angle = args[0][0]
        dist = args[0][1]

        x, y = to_cartesian([angle, dist])
        dx = dy = 1

        # Compute central difference for the x-component of the gradient
        x_plus = to_polar([x + dx, y])
        x_minus = to_polar([x - dx, y])

        grad_x = (self.func(np.array([x_plus[0], x_plus[1]])) -
                  self.func(np.array([x_minus[0], x_minus[1]]))) / (2 * dx)

        # Compute central difference for the y-component of the gradient
        y_plus = to_polar([x, y + dy])
        y_minus = to_polar([x, y - dy])

        grad_y = (self.func(np.array([y_plus[0], y_plus[1]])) -
                  self.func(np.array([y_minus[0], y_minus[1]]))) / (2 * dy)

        # Combine the gradients into an array
        grad = np.array([grad_x, grad_y])

        return grad


class RandomKeeperAgent(Agent):
    """
    An agent which chooses 10 random legal moves and performs the one which gives the lowest loss
    """

    def getAction(self, state):
        best_action = (0, 0)
        best_action_score = np.inf
        self.func = Func(state)
        for i in range(10):
            while True:
                angle = random.uniform(0.0, math.pi * 2)
                speed = random.uniform(100, 2000)
                val = self.func.compute(np.array([angle, speed]))
                if val < best_action_score:
                    best_action = (angle, speed)
                    best_action_score = val
                if state.do_action(best_action[0], best_action[1]).cue_ball is not None:
                    break
        return best_action


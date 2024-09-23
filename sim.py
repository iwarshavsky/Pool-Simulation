import sys, random
import time
import pygame
import pymunk.pygame_util

from utils.agent import *
from utils.helper import *

from utils.game_state import GameState

from utils.game_state import GameState

class Game:

    def __init__(self, agent=None, visible=False, ball_positions=None, cue_ball_position=None):

        """
        Run a new game with agent provided as input, choose whether the simulation should be visible or not
        :param agent: an instance of the Agent class
        :param visible: bool

        :param ball_positions: optional positions of non-cue-balls to start a game with
        :param cue_ball_position: optional position of cue_ball to start a game with
        :return: (T/F, num rounds, )
        """
        self.visible = visible
        self.agent = agent
        if self.agent is None:

            self.game_state = GameState(visible=True, ball_positions=ball_positions, cue_ball_position=cue_ball_position)
            self.visible = True
            self.agent = HumanAgent()
        else:
            self.game_state = GameState(visible=visible, ball_positions=ball_positions, cue_ball_position=cue_ball_position)

        self.rounds = 0

        self.is_game_over = False
        self.won = False
        self.rounds = 0

    def run(self):

        start_time = time.time()
        actions = []
        balls_removed_in_turn = []
        while not self.is_game_over:
            actions.append(self.agent.getAction(self.game_state))
            prev_number_of_balls = self.game_state.get_num_remaining_balls()
            self.game_state = self.game_state.do_action(*actions[-1], visible=self.visible) # unpacks angle and speed
            cur_number_of_balls = self.game_state.get_num_remaining_balls()
            balls_removed_in_turn.append(prev_number_of_balls - cur_number_of_balls)

            game_state_evaluation = GameState.evaluate(self.game_state)

            self.rounds += 1

            if game_state_evaluation is None:
                pass # can continue game
            elif game_state_evaluation:
                self.is_game_over = True
                self.won = True
            else:
                self.is_game_over = True
                self.won = False

        return self.won, self.rounds, sum(balls_removed_in_turn)/self.rounds, time.time() - start_time

def run_sim():
    Game().run() # Human control
    Game(agent=RandomAgent()).run() # Random agent, not visible
    Game(agent=RandomAgent(),visible=True).run()  # Random agent, not visible

if __name__ == '__main__':

    # random.seed(0)
    # Example runs:
    # Game().run() # Human control
    #Game(agent=RandomAgent(), visible=True).run() # Random agent, not visible
    Game(agent=LocalSearchAgent(debug=True),visible=True).run()  # Random agent, visible

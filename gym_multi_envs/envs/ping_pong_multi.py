""" This is a Multi agent version of PLE pong.py to support openAI Gym. This version has support for multi agent control.
original code:
https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/ple/games/pong.py
"""

import math
import os
import sys
import numpy as np

import pygame
from pygame.constants import K_w, K_s, K_UP, K_DOWN, K_F15, KEYDOWN

from .ping_pong import Ball, Player

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_multi_envs.envs.base import vec2d, PyGameWrapper



WIDTH = 720
HEIGHT = 480
FPS = 60

BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE  = (0,0,255)
RED  = (255,0,0)


def percent_round_int(percent, x):
    return np.round(percent * x).astype(int)


class PongMultiAgent(PyGameWrapper):
    """
    """
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, width=WIDTH, height=HEIGHT, player2_speed_ratio=0.6, player1_speed_ratio = 0.6, ball_speed_ratio=0.95,  MAX_SCORE=11):



        self.ball_radius = percent_round_int(height, 0.03)
        self.n_agents =2
        self.player2_speed_ratio = player2_speed_ratio
        self.ball_speed_ratio = ball_speed_ratio
        self.player1_speed_ratio = player1_speed_ratio

        self.paddle_width = percent_round_int(width, 0.023)
        self.paddle_height = percent_round_int(height, 0.15)
        self.paddle_dist_to_wall = percent_round_int(width, 0.0625)
        self.MAX_SCORE = MAX_SCORE

        self.a1_dy = 0.0
        self.a2_dy = 0.0
        self.score_sum = [0.0,0.0] # need to deal with 11 on either side winning
        self.score_counts = {
            "agent1": 0.0,
            "agent2": 0.0
        }
        self.rewards ={
        "positive":1.0,
        "negative":-1.0,
        "win": 10.0,
        "loss":-10,
        "tick":.0001
        }

        super().__init__(WIDTH, HEIGHT, FPS, True, True)
        if pygame.font:
            self.font = pygame.font.Font(None,30)
        else:
            self.font = None
        self.action_space = spaces.Tuple( [spaces.Discrete(3),spaces.Discrete(3)] )
        pygame.display.set_caption('PING PONG MULTI')
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, 3))

        # the %'s come from original values, wanted to keep same ratio when you
        # increase the resolution.



    def startState(self):
        self.score_counts = {
            "agent1": 0.0,
            "agent2": 0.0
        }

        self.rewards = {
        "positive":1.0,
        "negative":-1.0,
        "win": 10.0,
        "loss":-10,
        "tick":.00001
        }

        self.score_sum = [0.0,0.0]
        self.ball = Ball(
            self.ball_radius,
            self.ball_speed_ratio * self.height,
            self.np_random,
            (self.width / 2, self.height / 2),
            self.width,
            self.height,
            WHITE
        )

        self.agent1Player = Player(
            self.player1_speed_ratio * self.height,
            self.paddle_width,
            self.paddle_height,
            (self.paddle_dist_to_wall, self.height / 2),
            self.width,
            self.height,
            BLUE)

        self.agent2Player = Player(
            self.player2_speed_ratio * self.height,
            self.paddle_width,
            self.paddle_height,
            (self.width - self.paddle_dist_to_wall, self.height / 2),
            self.width,
            self.height,
            RED)
        self.agents = [self.agent1Player, self.agent2Player]

    def pygame_event_handler(self):
        for i, event in enumerate(pygame.event.get()):
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_w:
                    self.a1_dy = -self.agent1Player.speed

                if key == pygame.K_s:
                    self.a1_dy = self.agent1Player.speed

                if key == pygame.K_UP:
                    self.a2_dy = -self.agent2Player.speed

                if key == pygame.K_DOWN:
                    self.a2_dy = self.agent2Player.speed
        pygame.event.clear()

    def getGameState(self):
        """
        TODO change this whrn you add dict output
        Gets a non-visual state representation of the game.
        Returns
        -------
        dict
            * player y position.
            * players velocity.
            * cpu y position.
            * ball x position.
            * ball y position.
            * ball x velocity.
            * ball y velocity.
            See code for structure.
        """
        state = {
            "player1_y": self.agent1Player.pos.y,
            "player1_velocity": self.agent1Player.vel.y,
            "player2_y": self.agent2Player.pos.y,
            "player2_velocity": self.agent2Player.vel.y,
            "ball_x": self.ball.pos.x,
            "ball_y": self.ball.pos.y,
            "ball_velocity_x": self.ball.vel.x,
            "ball_velocity_y": self.ball.vel.y
        }

        return state

    def step(self, actions):

        prev_score = list(self.score_sum)

        done = [False, False]
        rewards = [0.0,0.0]
        ob = None
        for i, action in enumerate(actions):
            self.set_pyagme_events(action,self.agents[i])
            self.pygame_event_handler()
            self.score_sum[i] += self.rewards["tick"]

        if self.display_screen == True:
            self._draw_frame(self.display_screen)
        self.dt = self.tick(self.fps)

        dt = self.dt/1000.0

        self.screen.fill((0, 0, 0))

        self.agent1Player.speed = self.player1_speed_ratio * self.height
        self.agent2Player.speed = self.player2_speed_ratio * self.height
        self.ball.speed = self.ball_speed_ratio * self.height


        # doesnt make sense to have this, but include if needed.


        self.ball.update(self.agent1Player, self.agent2Player, dt)

        is_terminal_state = False

        # logic
        if self.ball.pos.x <= 0:
            self.score_sum[1] += self.rewards["negative"]
            self.score_counts["agent2"] += 1.0
            self._reset_ball(-1)
            is_terminal_state = True

        if self.ball.pos.x >= self.width:
            self.score_sum[0] += self.rewards["positive"]
            self.score_counts["agent1"] += 1.0
            self._reset_ball(1)
            is_terminal_state = True

        if is_terminal_state:
            # winning
            if self.score_counts['agent1'] == self.MAX_SCORE:
                self.score_sum[1] += self.rewards["win"]
                for i in range(len(done)):
                    done[i] = True

            # losing
            if self.score_counts['agent2'] == self.MAX_SCORE:
                self.score_sum[0] += self.rewards["loss"]
                for i in range(len(done)):
                    done[i] = True
        else:
            self.agent1Player.update(self.a1_dy, dt)
            self.agent2Player.update(self.a2_dy, dt)

        self.ball.draw(self.screen)
        self.agent1Player.draw(self.screen)
        self.agent2Player.draw(self.screen)
        self.show_stats()
        for i in range(len(prev_score)):
            rewards[i] = self.getScore()[i] - prev_score[i]

        ob = np.fliplr(np.rot90(self.getScreenRGB(),3))

        return ob, rewards, done, self.getGameState()

    def set_pyagme_events(self, action, agent):
        """Convert  gym action space to pygame events."""
        kd = None
        if action == 2:
            if agent is self.agent1Player:
                kd = pygame.event.Event(KEYDOWN, {"key": K_w})
            elif agent is self.agent2Player:
                kd = pygame.event.Event(KEYDOWN, {"key": K_UP})
        elif action == 0:
            if agent is self.agent1Player:
                kd = pygame.event.Event(KEYDOWN, {"key": K_s})
            elif agent is self.agent2Player:
                kd = pygame.event.Event(KEYDOWN, {"key": K_DOWN})
        else:
            kd = pygame.event.Event(KEYDOWN, {"key": K_F15})
        pygame.event.post(kd)

    def getScore(self):
        return self.score_sum

    def show_stats(self):
        if self.font:
            font_surface = self.font.render("CPU: " + str(self.score_counts['agent2']), False, WHITE)
            self.screen.blit(font_surface, (355,5))
            font_surface = self.font.render("Agent: " + str(self.score_counts['agent1']), False, WHITE)
            self.screen.blit(font_surface, (15,5))

    def reset(self):
        self.startState()
        # after game over set random direction of ball otherwise it will always be the same
        self._reset_ball(1 if self.np_random.random_sample() > 0.5 else -1)
        self._draw_frame(self.display_screen)

        ob = np.fliplr(np.rot90(self.getScreenRGB(),3))
        return ob


    def _reset_ball(self, direction):
        self.ball.pos.x = self.width / 2  # move it to the center

        # we go in the same direction that they lost in but at starting vel.
        self.ball.vel.x = self.ball.speed * direction
        self.ball.vel.y = (self.np_random.random_sample() *
                           self.ball.speed) - self.ball.speed * 0.5

import os
import sys
import numpy as np

import pygame
from pygame.constants import K_w, KEYDOWN, K_RIGHT, K_LEFT

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_multi_snake.envs import base

# Object dimensions
BRICK_WIDTH   = 60
BRICK_HEIGHT  = 15
PADDLE_WIDTH  = 60
PADDLE_HEIGHT = 12
BALL_DIAMETER = 16
BALL_RADIUS   = BALL_DIAMETER // 2


WIDTH = 360
HEIGHT = 480
FPS = 30



SCREEN_SIZE   = WIDTH,HEIGHT


MAX_PADDLE_X = SCREEN_SIZE[0] - PADDLE_WIDTH
MAX_BALL_X   = SCREEN_SIZE[0] - BALL_DIAMETER
MAX_BALL_Y   = SCREEN_SIZE[1] - BALL_DIAMETER

# Paddle Y coordinate
PADDLE_Y = SCREEN_SIZE[1] - PADDLE_HEIGHT - 10

# Color constants
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE  = (0,0,255)

# State constants
STATE_BALL_IN_PADDLE = 0
STATE_PLAYING = 1
STATE_WON = 2
STATE_GAME_OVER = 3


class BallPaddleGame(base.PyGameWrapper):
    """ Main game class that implements gym functions to control the game."""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, obs_type="Image"):

        self.obs_type = obs_type

        # preload assets

        super().__init__(WIDTH, HEIGHT, FPS, False, True)

        pygame.display.set_caption('basic')

        if pygame.font:
            self.font = pygame.font.Font(None,30)
        else:
            self.font = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = None
        if obs_type == "Image":
            self.screen_height, self.screen_width = self.getScreenDims()
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))
        else:
            self.observation_space = spaces.Dict({"ball": spaces.Tuple(spaces.Box(low=BALL_DIAMETER, high=MAX_BALL_X, shape=(1)),
                                                                    spaces.Box(low=BALL_DIAMETER, high=MAX_BALL_Y, shape=(1))),
                                                  "paddle": spaces.Tuple(spaces.Box(low=PADDLE_WIDTH, high=MAX_PADDLE_X, shape=(1)),
                                                                         spaces.Box(low=PADDLE_Y, high=PADDLE_Y, shape=(1)))})

    def startState(self):
        self.score = 0
        self.state = STATE_PLAYING

        self.paddle   = pygame.Rect(300,PADDLE_Y,PADDLE_WIDTH,PADDLE_HEIGHT)
        self.ball     = pygame.Rect(300,PADDLE_Y - BALL_DIAMETER,BALL_DIAMETER,BALL_DIAMETER)

        self.ball.left = self.paddle.left + self.paddle.width / 2
        self.ball.top  = self.paddle.top - self.ball.height

        self.ball_vel = [5,-5]

    def _move_ball(self):
        self.ball.left += self.ball_vel[0]
        self.ball.top  += self.ball_vel[1]

        if self.ball.left <= 0:
            self.ball.left = 0
            self.ball_vel[0] = -self.ball_vel[0]
        elif self.ball.left >= MAX_BALL_X:
            self.ball.left = MAX_BALL_X
            self.ball_vel[0] = -self.ball_vel[0]

        if self.ball.top < 0:
            self.ball.top = 0
            self.ball_vel[1] = -self.ball_vel[1]
        elif self.ball.top >= MAX_BALL_Y:
            self.ball.top = MAX_BALL_Y
            self.ball_vel[1] = -self.ball_vel[1]

    def _handle_collisions(self):

        if self.ball.colliderect(self.paddle):
            self.ball.top = PADDLE_Y - BALL_DIAMETER
            self.ball_vel[1] = -self.ball_vel[1]
            self.score += 1
        elif self.ball.top > self.paddle.top:
            self.state = STATE_GAME_OVER

    def show_stats(self):
        if self.font:
            font_surface = self.font.render("SCORE: " + str(self.score), False, WHITE)
            self.screen.blit(font_surface, (205,5))

    def step(self, action):
        self.screen.fill(BLACK)
        prev_score = self.score
        #self.check_input()
        super().step(action)
        done = False
        reward = 0
        ob = None
        if self.state == STATE_PLAYING:
            self._move_ball()
            self._handle_collisions()
            if prev_score < self.score:
                reward = 100

        if self.state == STATE_GAME_OVER:
            reward = -1000
            done = True

        # Draw paddle
        pygame.draw.rect(self.screen, BLUE, self.paddle)

        # Draw ball
        pygame.draw.circle(self.screen, WHITE, (self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS), BALL_RADIUS)

        self.show_stats()

        self._draw_frame(self.display_screen)

        if self.obs_type == "Image":
            ob = np.fliplr(np.rot90(self.getScreenRGB(),3))
        else:
            ob = {"ball":(self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS),
                  "paddle":(self.paddle.left,self.paddle.top)}

        return ob , reward, done, {}


    def set_pyagme_events(self, action):
        """Convert  gym action space to pygame events."""
        kd = None
        if action == 1:
            kd = pygame.event.Event(KEYDOWN, {"key": K_RIGHT})
        else:
            kd = pygame.event.Event(KEYDOWN, {"key": K_LEFT})
        pygame.event.post(kd)


    def pygame_event_handler(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_LEFT:
                    self.paddle.left -= 5
                    if self.paddle.left < 0:
                        self.paddle.left = 0

                if key==pygame.K_RIGHT:
                    self.paddle.left += 5
                    if self.paddle.left > MAX_PADDLE_X:
                        self.paddle.left = MAX_PADDLE_X

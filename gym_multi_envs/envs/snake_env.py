""" This Snake Gym Environment is based on https://github.com/EvolveSec/Snake-Multiplayer.git """
import os
import sys
import numpy as np
import random
import pygame
from pygame.constants import K_UP, KEYDOWN, K_RIGHT, K_LEFT, K_DOWN
from gym_multi_envs.envs import base

import gym
from gym import error, spaces, utils
from gym.utils import seeding

#create surface object
WIDTH = 720
HEIGHT = 480
FPS = 60

SCREEN_SIZE   = WIDTH,HEIGHT

#colours
GREEN = (0,255,0)
BLACK = (0,0,0)
CYAN = (175,238,238)
YELLOW = (255,255,0)
WHITE = (255,255,255)
RED = (255,0,0)
BLUE = (0,0,255)
GOLD = (255,255,153)
PURPLE = (147,112,219)
GREEN = (0,255,0)

class foodClass(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        #self.rng = rng
        #sets color and initial position of food
        self.image = pygame.Surface((20,20))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.newPos()

    def newPos(self):
        #sets new position of food
        self.possiblePosX = range(20, WIDTH, 20)
        self.possiblePosY = range(20, HEIGHT, 20)
        self.rect.x = random.choice(self.possiblePosX)
        self.rect.y = random.choice(self.possiblePosY)

    def update(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))

class blockClass(pygame.sprite.Sprite):

    def __init__(self, x, y, size, color):
        pygame.sprite.Sprite.__init__(self)
        #sets position, color, and size of block
        self.image = pygame.Surface((size,size))
        self.color = color
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def draw(self, surface):
        #draws blocks to surface
        surface.blit(self.image, (self.rect.x, self.rect.y))

class snakeClass(pygame.sprite.Sprite):

    def __init__(self, color, demo):
        pygame.sprite.Sprite.__init__(self)
        #creates snake object variables
        self.direction = 270
        self.body = []
        self.color = color
        self.dead = False
        self.score = 0
        self.demo = demo
        self.portals = []
        #creates initial snake body
        count = 0
        for i in range(15):
            self.body.append(blockClass(60 + count, 60, 20, self.color))
            count += 20

    def checkCollision(self, food):
        #checks if collision with wall
        if self.body[-1].rect.x > WIDTH:
            self.dead = True
        elif self.body[-1].rect.x < 0:
            self.dead = True
        elif self.body[-1].rect.y > HEIGHT:
            self.dead = True
        elif self.body[-1].rect.y < 0:
            self.dead = True

        #checks if collision with food
        if self.body[-1].rect.colliderect(food.rect):
            food.newPos()
            self.score += 1
            if self.demo:
                del self.body[0]
        else:
            del self.body[0]

        #checks if collision with own body
        for i in range(len(self.body[:-1])):
            if self.body[-1].rect.colliderect(self.body[i]):
                self.dead = True

    def intelligence(self):
        #intelligence method, moves to food depending on current direction
        if food.rect.y > self.body[-1].rect.y and self.direction != 0:
            self.direction = 180
        elif food.rect.y < self.body[-1].rect.y and self.direction != 180:
            self.direction = 0
        elif food.rect.x < self.body[-1].rect.x and self.direction != 270:
            self.direction = 90
        elif food.rect.x > self.body[-1].rect.x and self.direction != 90:
            self.direction = 270

    def resetValues(self):
        #reset values of snake
        pygame.sprite.Sprite.__init__(self)
        self.direction = 270
        self.body = []
        self.dead = False
        self.score = 0
        self.portals = []

        count = 0
        for i in range(15):
            self.body.append(blockClass(60 + count, 60, 20, self.color))
            count += 20

    def update(self, surface, food):
        #get coordinates of front position of snake
        x_frontPos = self.body[-1].rect.x
        y_frontPos = self.body[-1].rect.y

        #update snake depending on direction
        if self.direction == 180:
            self.body.append(blockClass(x_frontPos, y_frontPos + 20, 20, self.color))
        elif self.direction == 0:
            self.body.append(blockClass(x_frontPos, y_frontPos - 20, 20, self.color))
        elif self.direction == 90:
            self.body.append(blockClass(x_frontPos - 20, y_frontPos, 20, self.color))
        elif self.direction == 270:
            self.body.append(blockClass(x_frontPos + 20, y_frontPos, 20, self.color))
        #check snake collisions
        self.checkCollision(food)
        #update snake on screen
        for i in range(len(self.body)):
            self.body[i].image.set_alpha(i*10)
            self.body[i].draw(surface)
        #update portals on screen
        for portal in self.portals:
            portal.update(surface)
        #if demo snake, activate A.I
        if self.demo:
            self.intelligence()

class SnakeGame(base.PyGameWrapper):
    """ Main game class that implements gym functions to control the game."""
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, width=WIDTH, height=HEIGHT):

        self.snake = snakeClass(CYAN, False)
        self.food = foodClass()

        super().__init__(WIDTH, HEIGHT, fps=FPS)
        if pygame.font:
            self.font = pygame.font.Font(None, 30)
        else:
            self.font = None

        self.action_space = spaces.Discrete(4)

        #super()._draw_frame(self.display_screen)
        pygame.display.set_caption('Single Snake')
        self.screen_height, self.screen_width = self.getScreenDims()
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))

    def startState(self):
        self.score = 0
        self.snake.resetValues()
        self.food.newPos()

    def show_stats(self):
        if self.font:
            text = self.font.render('Player 1 score:' + str(self.snake.score), True, WHITE)
            self.screen.blit(text, (5,5))

    def _draw_frame(self, draw_screen):
        super()._draw_frame(draw_screen)

        self.screen.fill(BLACK)
        self.show_stats()

    def step(self, action):
        prev_score = self.score
        done = False
        reward = 0
        ob = None
        super().step(action)
        self.food.update(self.screen)
        self.snake.update(self.screen, self.food)

        if self.snake.dead:
            done = True

        self.score =self.snake.score
        reward = self.score - prev_score
        reward += .001

        self._draw_frame(self.display_screen)
        ob = np.fliplr(np.rot90(self.getScreenRGB(),3))

        return ob , reward, done, {}

    def reset(self):
        super().reset()
        self._draw_frame(self.display_screen)
        ob = np.fliplr(np.rot90(self.getScreenRGB(),3))
        return ob

    def set_pyagme_events(self, action):
        """Convert  gym action space to pygame events."""
        kd = None
        if action == 0:
            kd = pygame.event.Event(KEYDOWN, {"key": K_RIGHT})
        elif action == 2:
            kd = pygame.event.Event(KEYDOWN, {"key": K_LEFT})
        elif action == 1:
            kd = pygame.event.Event(KEYDOWN, {"key": K_DOWN})
        elif action == 3:
            kd = pygame.event.Event(KEYDOWN, {"key": K_UP})
        pygame.event.post(kd)

    def pygame_event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_DOWN:
                    if self.snake.direction != 0:
                        self.snake.direction = 180
                elif key == pygame.K_UP:
                    if self.snake.direction != 180:
                        self.snake.direction = 0
                elif key == pygame.K_LEFT:
                    if self.snake.direction != 270:
                        self.snake.direction = 90
                elif key == pygame.K_RIGHT:
                    if self.snake.direction != 90:
                        self.snake.direction = 270

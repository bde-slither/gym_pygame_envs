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
FPS = 1000

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
MAX_SCORE = 35
MAX_STEPS = 2000
FOOD_COUNT = 5

class foodClass(pygame.sprite.Sprite):

    def __init__(self,rng):
        pygame.sprite.Sprite.__init__(self)
        self.rng = rng
        #sets color and initial position of food

        self.rect = pygame.Rect(0, 0, 20 , 20)
        self.newPos()

    def newPos(self):
        #sets new position of food
        self.possiblePosX = range(20, WIDTH-20, 20)
        self.possiblePosY = range(20, HEIGHT-20, 20)
        self.rect.x = self.rng.choice(self.possiblePosX)
        self.rect.y = self.rng.choice(self.possiblePosY)

    def draw(self,screen):
        pygame.draw.rect(screen, WHITE, self.rect)

class blockClass(object):

    def __init__(self, x, y, size):
        #sets position, color, and size of block
        self.rect = pygame.Rect(0, 0, size , size)
        self.rect.x = x
        self.rect.y = y

    def draw(self,screen, color):
        pygame.draw.rect(screen, color, self.rect)

class snakeClass(object):

    def __init__(self, color):
        #creates snake object variables
        self.direction = 270
        self.body = []
        self.color = color
        self.dead = False
        self.score = 0
        self.portals = []
        #creates initial snake body
        count = 0
        for i in range(10):
            self.body.append(blockClass(60 + count, 60, 20))
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
        hasCollided = False
        for f in food:
            if self.body[-1].rect.colliderect(f.rect):
                f.newPos()
                self.score += 1
                hasCollided = True
        if not hasCollided:
            del self.body[0]

        #checks if collision with own body
        for i in range(len(self.body[:-1])):
            if self.body[-1].rect.colliderect(self.body[i]):
                self.dead = True

    def resetValues(self):
        #reset values of snake
        self.direction = 270
        self.body = []
        self.dead = False
        self.score = 0
        self.portals = []

        count = 0
        for i in range(10):
            self.body.append(blockClass(60 + count, 60, 20))
            count += 20

    def update(self, surface, food):
        #get coordinates of front position of snake
        x_frontPos = self.body[-1].rect.x
        y_frontPos = self.body[-1].rect.y

        #update snake depending on direction
        if self.direction == 180:
            self.body.append(blockClass(x_frontPos, y_frontPos + 20, 20))
        elif self.direction == 0:
            self.body.append(blockClass(x_frontPos, y_frontPos - 20, 20))
        elif self.direction == 90:
            self.body.append(blockClass(x_frontPos - 20, y_frontPos, 20))
        elif self.direction == 270:
            self.body.append(blockClass(x_frontPos + 20, y_frontPos, 20))
        #check snake collisions
        self.checkCollision(food)
        #update snake on screen
        for i in range(len(self.body)):
            color = self.color
            self.body[i].draw(surface, color)

class SnakeGame(base.PyGameWrapper):
    """ Main game class that implements gym functions to control the game."""
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.steps =0
        super().__init__(WIDTH, HEIGHT, fps=FPS,force_fps=True)
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
        self.steps =0
        self.snake = snakeClass(CYAN)
        self.food = []
        for i in range(0, FOOD_COUNT):
            self.food.append(foodClass(self.np_random))
        self.score = 0
        self.snake.resetValues()
        for f in self.food:
            f.newPos()

    def show_stats(self):
        if self.font:
            text = self.font.render('Player 1 score:' + str(self.snake.score), True, WHITE)
            self.screen.blit(text, (5,5))

    def step(self, action):
        prev_score = self.score
        done = False
        reward = 0
        ob = None
        super().step(action)
        self.screen.fill(BLACK)
        self.steps +=1
        for f in self.food:
            f.draw(self.screen)
        self.snake.update(self.screen, self.food)
        self.show_stats()

        if self.snake.dead:
            done = True

        self.score =self.snake.score
        reward = self.score - prev_score
        reward += .001

        pygame.display.update()
        ob = np.fliplr(np.rot90(self.getScreenRGB(),3))
        if self.steps>=MAX_STEPS or self.score>=MAX_SCORE:
            done=True

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

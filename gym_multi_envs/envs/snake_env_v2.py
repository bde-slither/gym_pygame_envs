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
WIDTH = 0
HEIGHT = 0
FPS = 0

SCREEN_SIZE   = 0
BLOCK_WIDTH = 5
INIT_OFFSET = 20
SNAKE_COUNT = 0
MAX_SCORE = 0
MAX_STEP = 0

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

FOOD_COUNT = 10
SNAKE_LENGTH = 10

CROP_OFFSET = 10
CROP_WIDTH = 180
CROP_HEIGHT = 120

class foodClass(pygame.sprite.Sprite):

    def __init__(self,rng):
        pygame.sprite.Sprite.__init__(self)
        self.rng = rng
        #sets color and initial position of food

        self.rect = pygame.Rect(0, 0, BLOCK_WIDTH , BLOCK_WIDTH)
        self.newPos()

    def newPos(self):
        global WIDTH
        global HEIGHT

        #sets new position of food
        self.possiblePosX = range(BLOCK_WIDTH, WIDTH-BLOCK_WIDTH, BLOCK_WIDTH)
        self.possiblePosY = range(BLOCK_WIDTH, HEIGHT-BLOCK_WIDTH, BLOCK_WIDTH)
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
        self.prevScore = 0
        self.portals = []
        self.cropLX = 0
        self.cropLY = 0

        #creates initial snake body
        self.resetValues()

    def checkCollision(self, food, idx, snake):
        if self.dead:
            return

        global WIDTH
        global HEIGHT

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
        
        for i, s in enumerate(snake):
            if i != idx:
                for b in s.body:
                    if self.body[-1].rect.colliderect(b):
                        self.dead = True
                        break
                if self.dead:
                    break

        if self.dead:
            self.body = []

    def resetValues(self):
        #reset values of snake
        self.direction = 270
        self.body = []
        self.dead = False
        self.score = 0
        self.prevScore = 0
        self.portals = []

        startX = int(random.random() * (WIDTH - 2 * INIT_OFFSET) + INIT_OFFSET)
        startY = int(random.random() * (HEIGHT - 2 * INIT_OFFSET) + INIT_OFFSET)
        
        count = 0
        for i in range(SNAKE_LENGTH):
            self.body.append(blockClass(startX + count, startY, BLOCK_WIDTH))
            count += BLOCK_WIDTH

        self.cropLX = self.body[-1].rect.x
        self.cropLY = self.body[-1].rect.y
        self.reAlignCropXY()
        
    def update(self, surface, food, idx, snake):
        if self.dead:
            return
        
        #print (len(self.body), len(food))
        #get coordinates of front position of snake
        x_frontPos = self.body[-1].rect.x
        y_frontPos = self.body[-1].rect.y

        #update snake depending on direction
        if self.direction == 180:
            self.body.append(blockClass(x_frontPos, y_frontPos + BLOCK_WIDTH, BLOCK_WIDTH))
        elif self.direction == 0:
            self.body.append(blockClass(x_frontPos, y_frontPos - BLOCK_WIDTH, BLOCK_WIDTH))
        elif self.direction == 90:
            self.body.append(blockClass(x_frontPos - BLOCK_WIDTH, y_frontPos, BLOCK_WIDTH))
        elif self.direction == 270:
            self.body.append(blockClass(x_frontPos + BLOCK_WIDTH, y_frontPos, BLOCK_WIDTH))
        #check snake collisions
        self.checkCollision(food, idx, snake)
        if self.dead:
            return
        
        #update snake on screen
        for i in range(len(self.body)):
            color = self.color
            self.body[i].draw(surface, color)
        
        global WIDTH
        global HEIGHT
        global CROP_WIDTH
        global CROP_HEIGHT

        x = self.body[-1].rect.x
        y = self.body[-1].rect.y
        changed = False
        if x - self.cropLX <= CROP_OFFSET or self.cropLX + CROP_WIDTH - x - BLOCK_WIDTH < CROP_OFFSET:
            changed = True
            self.cropLX = self.body[-1].rect.x - int(CROP_WIDTH / 2)
        if y - self.cropLY <= CROP_OFFSET or self.cropLY + CROP_HEIGHT - y - BLOCK_WIDTH < CROP_OFFSET:
            changed = True
            self.cropLY = self.body[-1].rect.y - int(CROP_HEIGHT / 2)

        if changed:
            self.reAlignCropXY()

    def reAlignCropXY(self):
        if self.cropLX < 0:
            self.cropLX = 0
        elif self.cropLX >= WIDTH:
            self.cropLX = WIDTH - CROP_WIDTH
        if self.cropLY < 0:
            self.cropLY = 0
        elif self.cropLY >= HEIGHT:
            self.cropLY = HEIGHT - CROP_HEIGHT
        
        if self.cropLX + CROP_WIDTH >= WIDTH:
            self.cropLX -= (self.cropLX + CROP_WIDTH - WIDTH)
        if self.cropLY + CROP_HEIGHT >= HEIGHT:
            self.cropLY -= (self.cropLY + CROP_HEIGHT - HEIGHT)

class SnakeGameV2(base.PyGameWrapper):
    """ Main game class that implements gym functions to control the game."""
    metadata = {'render.modes': ['human', 'rgb_array']}
    n_agents = 0

    def __init__(self, **kwargs):

        print (kwargs)
        #for key, value in kwargs.iteritems():
        #    print ("%s = %s" % (key, value))
        self.stepCount = 0
        global WIDTH
        global HEIGHT
        global FPS
        global SCREEN_SIZE
        global SNAKE_COUNT
        global FOOD_COUNT
        global MAX_SCORE
        global MAX_STEP

        WIDTH = kwargs['WIDTH']
        HEIGHT = kwargs['HEIGHT']
        FPS = kwargs['FPS']
        SNAKE_COUNT = kwargs['SNAKE_COUNT']
        FOOD_COUNT = kwargs['FOOD_COUNT']
        MAX_SCORE = kwargs['MAX_SCORE']
        MAX_STEP = kwargs['MAX_STEP']

        self.n_agents = SNAKE_COUNT

        SCREEN_SIZE = WIDTH * HEIGHT

        #super().__init__(WIDTH, HEIGHT, fps=FPS,force_fps=True)
        super().__init__(WIDTH, HEIGHT, fps=FPS,force_fps=False)
        
        if pygame.font:
            self.font = pygame.font.Font(None, 30)
        else:
            self.font = None

        self.action_space = spaces.Discrete(4)

        #super()._draw_frame(self.display_screen)
        pygame.display.set_caption('Single Snake')
        self.screen_height, self.screen_width = self.getScreenDims()
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))

    def render(self, mode='human'):
        """
        This method render scenes taken from pygame.
        """
        self._draw_frame(self.display_screen)
        
        subSurface = pygame.display.get_surface().subsurface(pygame.Rect(self.snake[0].cropLX, self.snake[0].cropLY, CROP_WIDTH, CROP_HEIGHT))
        screenRGB = pygame.surfarray.array3d(subSurface).astype(np.uint8)

        if mode == 'rgb_array':
            return np.fliplr(np.rot90(screenRGB,3))# return RGB frame suitable for video
        elif mode is 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                img = np.fliplr(np.rot90(screenRGB,3))
                self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
        else:
            raise ValueError

    def startState(self):
        self.snake = []
        for i in range(0, SNAKE_COUNT):
            s = snakeClass(CYAN)
            s.resetValues()
            self.snake.append(s)
        
        self.food = []
        for i in range(0, FOOD_COUNT):
            self.food.append(foodClass(self.np_random))
        
        for f in self.food:
            f.newPos()

    def show_stats(self):
        if self.font:
            stats = []
            for idx, s in enumerate(self.snake):
                stats.append('P' + str(idx) + ': ' + str(s.score))
            
            text = self.font.render(' | '.join(stats), True, WHITE)
            self.screen.blit(text, (5,5))

    def step(self, action):
        ob = None
        #super().step(action)
        for idx, s in enumerate(self.snake):
            if self.action_space.contains(action[idx]):
                #self.set_pyagme_events(action)
                if action[idx] == 1:
                    if s.direction != 0:
                        s.direction = 180
                elif action[idx] == 3:
                    if s.direction != 180:
                        s.direction = 0
                elif action[idx] == 2:
                    if s.direction != 270:
                        s.direction = 90
                elif action[idx] == 0:
                    if s.direction != 90:
                        s.direction = 270
            else:
                raise TypeError("action not in Action space.")
        
        #self.pygame_event_handler()
        if self.display_screen == True:
            self._draw_frame(self.display_screen)
        self.dt = self.tick(self.fps)

        self.screen.fill(BLACK)
        for f in self.food:
            f.draw(self.screen)
        
        global MAX_SCORE
        global MAX_STEP

        doneOverride = False
        done = []
        reward = []
        for idx, s in enumerate(self.snake):
            s.update(self.screen, self.food, idx, self.snake)
            reward.append(s.score - s.prevScore + 0.001)
            s.prevScore = s.score
            done.append(s.dead)
            if s.score == MAX_SCORE:
                doneOverride = True

        if self.stepCount == MAX_STEP:
            doneOverride = True

        self.show_stats()
        #print ('out', done)
        
        global CROP_HEIGHT
        global CROP_WIDTH
        surface = pygame.display.get_surface()

        ob = []
        for idx, s in enumerate(self.snake):
            for i, sn in enumerate(self.snake):
                #update snake on screen
                if idx == i:
                    for b in range(len(sn.body)):
                        color = sn.color
                        sn.body[b].draw(surface, color)
                else:
                    for b in range(len(sn.body)):
                        color = RED
                        sn.body[b].draw(surface, color)
            
            if self.display_screen == True:
                self._draw_frame(self.display_screen)

            #print (s.cropLX, s.cropLY, s.cropLX + CROP_WIDTH, s.cropLY + CROP_HEIGHT)
            subSurface = pygame.display.get_surface().subsurface(pygame.Rect(s.cropLX, s.cropLY, CROP_WIDTH, CROP_HEIGHT))
            #pygame.image.save(subSurface, 'sn_' + str(idx) + '_' + str(self.stepCount) + '.png')
            screenRGB = pygame.surfarray.array3d(subSurface).astype(np.uint8)
            img = np.fliplr(np.rot90(screenRGB,3))
            ob.append(img)
        self.stepCount += 1
        return ob, reward, done, {}

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
        if len(self.snake) == 0:
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_DOWN:
                    if self.snake[0].direction != 0:
                        self.snake[0].direction = 180
                elif key == pygame.K_UP:
                    if self.snake[0].direction != 180:
                        self.snake[0].direction = 0
                elif key == pygame.K_LEFT:
                    if self.snake[0].direction != 270:
                        self.snake[0].direction = 90
                elif key == pygame.K_RIGHT:
                    if self.snake[0].direction != 90:
                        self.snake[0].direction = 270

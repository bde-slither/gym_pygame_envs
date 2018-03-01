import os
import sys
import numpy as np

import pygame
from pygame.constants import K_w
from gym_multi_snake.envs import base


WIDTH = 360
HEIGHT = 480
FPS = 30

class SnakeGame(base.PyGameWrapper):
    """ Main game class that implements gym functions to control the game."""

    def __init__(self, width=WIDTH, height=HEIGHT):
        actions = {
            "up" : K_w
        }
        fps = FPS

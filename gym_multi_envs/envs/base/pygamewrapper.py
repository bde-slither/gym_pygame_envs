"""This PyGame wrapper for openAI gym is modified version of pygame wrapper
from PLE repo: https://github.com/ntasfi/PyGame-Learning-Environment"""
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from subprocess import call

import pygame
import numpy as np
from pygame.constants import KEYDOWN, KEYUP, K_F15

#os.environ["SDL_VIDEODRIVER"] = "directx"
os.environ["SDL_VIDEODRIVER"] = "X11"
#os.environ["SDL_VIDEODRIVER"] = "dummy"

class PyGameWrapper(gym.Env):
    """PyGameWrapper  class

    ple.games.base.PyGameWrapper(width, height, actions={})

    This :class:`PyGameWrapper` class sets methods all games require. It should be subclassed when creating new games.

    Parameters
    ----------
    width: int
        The width of the game screen.

    height: int
        The height of the game screen.

    actions: dict
        Contains possible actions that the game responds too. The dict keys are used by the game, while the values are `pygame.constants` referring the keys.

        Possible actions dict:

        >>> from pygame.constants import K_w, K_s
        >>> actions = {
        >>>     "up": K_w,
        >>>     "down": K_s
        >>> }
    """

    def __init__(self, width, height, fps=30, force_fps=False, display_screen=True):
        """Call super for this function in child class."""
        # Required fields
        self.rng = None
        self.action_space = None  # holds actions
        self.height = height
        self.width = width
        self.screen = None
        self.clock = None
        self.dt = 0
        self.screen_dim = (width, height)  # width and height
        self.display_screen = display_screen

        self.force_fps = force_fps
        self.fps = fps  # fps that the game is allowed to run at.
        self.NOOP = K_F15  # the noop key

        # intializing viwer for rendering.
        self.viewer = None

        self.seed()
        # setup PyGame
        pygame.init()
        call(["python", "--version"])
        pygame.display.init()


        # initialize the Game, raise error if not implemented.
        self.startState()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def startState(self):
        """
        This is used to initialize the game, such reseting the score, lives, and player position.

        This is game dependent.

        """
        raise NotImplementedError("Please override this method")

    def getScreenRGB(self):
        """
        Returns the current game screen in RGB format.

        Returns
        --------
        numpy uint8 array
            Returns a numpy array with the shape (width, height, 3).

        """

        #surf = pygame.display.get_surface()
        #print ("surf", type(surf))
        #r = pygame.Rect(0,0,10,10)
        #subSurf = surf.subsurface((r))
        #print ("subSurf", type(subSurf))
        return pygame.surfarray.array3d(
            pygame.display.get_surface()).astype(np.uint8)

    def _draw_frame(self, draw_screen):
        """
        Decides if the screen will be drawn too
        """
        if self.screen == None or self.clock==None:
                self.screen = pygame.display.set_mode(self.getScreenDims(),pygame.DOUBLEBUF|pygame.SRCALPHA , 32)
                self.screen.set_alpha(None)
                self.clock = pygame.time.Clock()
        pygame.display.flip()

    def tick(self, fps):
        """
        This sleeps the game to ensure it runs at the desired fps.
        """
        if self.force_fps:
            return 1000.0 / self.fps
        else:
            return self.clock.tick_busy_loop(fps)

    def step(self, action):
        """Call super when overriding this function if you want to draw game screen
        at each step after taking desired actions.
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if self.action_space.contains(action):
            self.set_pyagme_events(action)
        else:
            raise TypeError("action not in Action space.")
        self.pygame_event_handler()
        if self.display_screen == True:
            self._draw_frame(self.display_screen)
        self.dt = self.tick(self.fps)
        return None

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        self.startState()

    def render(self, mode='human'):
        """
        This method render scenes taken from pygame.
        """
        self._draw_frame(self.display_screen)
        if mode == 'rgb_array':
                    return np.fliplr(np.rot90(self.getScreenRGB(),3))# return RGB frame suitable for video
        elif mode is 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                img = np.fliplr(np.rot90(self.getScreenRGB(),3))
                self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
        else:
            raise ValueError

    def close(self):
        pygame.quit()
        if self.viewer: self.viewer.close()

    def set_pyagme_events(self, action):
        """Convert  gym action space to pygame events."""
        raise NotImplementedError

    def pygame_event_handler(self):
        """Define the how the game behaves for a given action."""
        raise NotImplementedError

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        Returns
        -------
        dict or None
            dict if the game supports it and None otherwise.

        """
        return None

    def getScreenDims(self):
        """
        Gets the screen dimensions of the game in tuple form.

        Returns
        -------
        tuple of int
            Returns tuple as follows (width, height).

        """
        return self.screen_dim

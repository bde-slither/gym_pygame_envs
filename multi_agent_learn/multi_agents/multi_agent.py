""" Based on keras rl base librabry."""
import os
import warnings
from copy import deepcopy

import numpy as np

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *

from keras.callbacks import History
from keras.callbacks import TensorBoard

from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class IndieMultiAgent(Agent):

    def __init__(self, agents):
        if len(agents)<2:
            raise RuntimeError('you have either one or zero agents. Please use Base keras-rl agents.')
        for agent in agents:
            if not agent.compiled:
                raise RuntimeError('you are agents are not compiled yet. Please call `compile()` before `fit()`.')
        self.agents = agents
        self.training = False
        self.step = 0
        self.compiled = False
        self.processor =agents[0].processor
        self.reset_states()

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        """Trains the agent on the given environment.
        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.
        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """

        self.training = True
        callback_multi=callbacks
        for agent in self.agents:
            agent.training = True
            callback_multi[agent]=CallbackList(callback_multi[agent])
            callbacks=CallbackList(callback_multi[agent])
        if hasattr(callbacks, 'set_model'):
            for agent in self.agents:
                callback_multi[agent].set_model(agent)
        else:
            for agent in self.agents:
                callback_multi[agent]._set_model(agent)
        for agent in self.agents:
            callback_multi[agent]._set_env(env)

        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            for agent in self.agents:            
                callback_multi[agent].set_params(params)
        else:
            for agent in self.agents:            
                callback_multi[agent]._set_params(params)
        self._on_train_begin()
        for agent in self.agents:            
                callback_multi[agent].on_train_begin()

        episode = 0
        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    for agent in self.agents:
                        #print("Parent")
                        #print(agent)
                        #print(callback_multi[agent])
                        callback_multi[agent].on_episode_begin(episode)
                    episode_step = 0
                    episode_reward = []
                    for agent in self.agents:
                        episode_reward.append(0.)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.processor is not None:
                            action = self.processor.process_action(action)

                        for agent in self.agents:
                            callback_multi[agent].on_action_begin(action)
                        observation, reward, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
                        for agent in self.agents:
                            callback_multi[agent].on_action_end(action)

                        if True in  done:
                            warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = self.processor.process_observation(observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                for agent in self.agents:
                    callback_multi[agent].on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward =[]
                done =[]
                for agent in self.agents:
                    reward.append(0.)
                    done.append(False)
                accumulated_info = {}
                for _ in range(action_repetition):
                    for agent in self.agents:
                        callback_multi[agent].on_action_begin(action)
                    observation, r, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(observation, r, done, info)


                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    for agent in self.agents:
                        callback_multi[agent].on_action_end(action)
                    reward = (np.array(reward)+np.array(r)).tolist()
                    if True in  done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    for idx, agent in enumerate(self.agents):
                        done[idx] = True
                
                metrics = self.backward(reward, terminals=done)
                episode_reward = (np.array(episode_reward)+np.array(reward)).tolist()
                step_logs={}
                for i,agent in enumerate(self.agents):
                    step_logs[agent]={
                        'action': action,
                        'observation': observation,
                        'reward': reward[i],
                        'metrics': metrics[i],
                        'episode': episode,
                        'info': accumulated_info,
                    }
                for agent in self.agents:
                    callback_multi[agent].on_step_end(episode_step, step_logs[agent])
                episode_step += 1
                self.step+=1
                for agent in self.agents:
                    agent.step += 1

                if True in done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    done =[]
                    for agent in self.agents:
                        done.append(False)
                    self.backward([0.]*len(self.agents), terminals=done)

                    # This episode is finished, report and reset.
                    episode_logs={}
                    for i,agent in enumerate(self.agents):
                        episode_logs[agent]= {
                            'episode_reward': episode_reward[i],
                            'nb_episode_steps': episode_step,
                            'nb_steps': self.step,
                        }
                    for agent in self.agents:
                        callback_multi[agent].on_episode_end(episode, episode_logs[agent])

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        for agent in self.agents:
            callback_multi[agent].on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        
        #return history

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1,agents=None):
        """Callback that is called before training begins."
        """
        self.agents=agents
        print (self.agents)
        self.training = False
        self.step = 0
        callback_multi=callbacks
        #print(callback_multi[agents])
        for agent in self.agents:
            agent.training = False
            callback_multi[agent]=CallbackList(callback_multi[agent])
            callbacks=CallbackList(callback_multi[agent])
        if hasattr(callbacks, 'set_model'):
            for agent in self.agents:
                callback_multi[agent].set_model(agent)
        else:
            for agent in self.agents:
                callback_multi[agent]._set_model(agent)
        for agent in self.agents:
            callback_multi[agent]._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            for agent in self.agents:
                callback_multi[agent].set_params(params)
        else:
            for agent in self.agents:
                callback_multi[agent]._set_params(params)

        self._on_test_begin()
        for agent in self.agents:
            callback_multi[agent].on_train_begin()
        for episode in range(nb_episodes):
            for agent in self.agents:
                callback_multi[agent].on_episode_begin(episode)
            episode_reward = []
            for agent in self.agents:
                episode_reward.append(0.)
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)

                for agent in self.agents:
                    callback_multi[agent].on_action_begin(action)
                observation, reward, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
                for agent in self.agents:
                    callback_multi[agent].on_action_end(action)

                if True in  done:
                    warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    break

            # Run the episode until we're done.
            done =[]
            for agent in self.agents:
                done.append(False)
            while not (True in done):
                for agent in self.agents:
                    callback_multi[agent].on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward =[]
                for agent in self.agents:
                    reward.append(0.)
                accumulated_info = {}
                for _ in range(action_repetition):
                    for agent in self.agents:
                        callback_multi[agent].on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(observation, r, d, info)
                    for agent in self.agents:
                        callback_multi[agent].on_action_end(action)
                    reward = (np.array(reward)+np.array(r)).tolist()
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if False in  done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    for idx, agent in enumerate(self.agents):
                        done[idx] = True

                self.backward(reward, terminals=done)
                episode_reward = (np.array(episode_reward)+np.array(reward)).tolist()

                step_logs={}
                for i,agent in enumerate(self.agents):
                    step_logs[agent]={
                        'action': action,
                        'observation': observation,
                        'reward': reward[i],
                        #'metrics': metrics[i],
                        'episode': episode,
                        'info': accumulated_info,
                    }
                for agent in self.agents:
                    callback_multi[agent].on_step_end(episode_step, step_logs[agent])
                episode_step += 1
                self.step+=1
                for agent in self.agents:
                    agent.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            done =[]
            for agent in self.agents:
                done.append(False)
            self.backward([0.]*len(self.agents), terminals=done)

            # Report end of episode.
            episode_logs={}
            for i,agent in enumerate(self.agents):
                episode_logs[agent]= {
                    'episode_reward': episode_reward[i],
                    'nb_episode_steps': episode_step,
                    'nb_steps': self.step,
                    }
            for agent in self.agents:
                callback_multi[agent].on_episode_end(episode, episode_logs[agent])
        for agent in self.agents:
            callback_multi[agent].on_train_end()
        self._on_test_end()

        #return history

    def compile(self, optimizer, metrics=[]):
        pass

    def forward(self, observation):
        actions = ()
        for idx, agent in enumerate(self.agents):
            actions+=(agent.forward(observation),)
        return actions
    def backward(self, rewards, terminals):
        list_metrics = []
        #print(terminals)
        for idx, agent in enumerate(self.agents):
            list_metrics.append(agent.backward(rewards[idx], terminals[idx]))
        return list_metrics

    def reset_states(self):
        for agent in self.agents:
            agent.reset_states()
    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        for idx, agent in enumerate(self.agents):
            agent_filepath = filename+"_agent"+str(idx)+extension
            agent.save_weights(agent_filepath)
    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        for idx, agent in enumerate(self.agents):
            agent_filepath = filename+"_agent"+str(idx)+extension
            agent.load_weights(agent_filepath)

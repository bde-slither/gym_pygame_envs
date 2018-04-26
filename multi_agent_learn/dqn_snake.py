from __future__ import division
import argparse
from time import gmtime, strftime
from PIL import Image
import numpy as np
import gym
import gym_multi_envs
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from keras.callbacks import TensorBoard

from multi_agent_learn.multi_agents import IndieMultiAgent
from keras.callbacks import History

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observations):
        obs = []
        for observation in observations:
            assert observation.ndim == 3  # (height, width, channel)
            img = Image.fromarray(observation)
            img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
            processed_observation = np.array(img)
            assert processed_observation.shape == INPUT_SHAPE
            obs.append(processed_observation.astype('uint8'))
        return obs[0]  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, rewards):
        r =[]
        for reward in rewards:
            r.append(np.clip(reward, -5., 5.))
        return r

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='ball_paddle-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
if args.env_name == "ball_paddle-v0":
    env.set_obs_type("Image")
np.random.seed(123)
env.seed(123)
nb_agents = env.n_agents

# List to hold DQN agents for this envs.
agents = []

for idx in range(nb_agents):
    nb_actions = env.action_space.n
    # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print("Player "+str((idx+1))+" summary")
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.2, value_min=.1, value_test=.05,
                                  nb_steps=1500000)

    # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # is Boltzmann-style exploration:
    #policy = BoltzmannQPolicy(tau=1.)
    # Feel free to give it a try!

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, batch_size=32,enable_dueling_network=True, dueling_type='avg',
                   processor=processor, nb_steps_warmup=20000, gamma=.99, target_model_update=10000,
                   train_interval=1, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    #dqn.load_weights("duelingqn2_snake-v0_weights_3000000.h5f")
    agents.append(dqn)

mdqn = IndieMultiAgent(agents)
callback_multi_test={}
for id,agent in enumerate(agents):
        callbacks = [(TestLogger())]
        history=History()
        callbacks += [history]
        callback_multi_test[agent]=callbacks
if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    callback_multi={}
    log_interval=10000
    for id,agent in enumerate(agents):
        weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
        checkpoint_weights_filename = 'dqn_' + args.env_name+'_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(args.env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        tensorboard = TensorBoard(log_dir="logs/{}_ESP_Greedy_"+"agent"+str(id)+"{}".format( args.env_name , strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        callbacks += [FileLogger(log_filename, interval=100000), tensorboard,TrainIntervalLogger(interval=log_interval)]
        history=History()
        callbacks += [history]
        callback_multi[agent]=callbacks
    mdqn.load_weights("dqn_snake-v2_weights.h5f")
    mdqn.fit(env, callbacks=callback_multi, nb_steps=2500000, log_interval=log_interval)

    # After training is done, we save the final weights one more time.
    mdqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    mdqn.test(env, nb_episodes=10,callbacks=callback_multi_test, visualize=False,agents=agents)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    mdqn.load_weights(weights_filename)
    mdqn.test(env, callbacks=callback_multi_test,nb_episodes=100, visualize=False,agents=agents)

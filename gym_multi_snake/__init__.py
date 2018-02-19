from gym.envs.registration import register

register(
    id='multi_snake-v0',
    entry_point='gym_multi_snake.envs:SnakeGame',
)

from gym.envs.registration import register

register(
    id='ball_paddle-v0',
    entry_point='gym_multi_envs.envs:BallPaddleGame',
)
register(
    id='ping_pong_single-v0',
    entry_point='gym_multi_envs.envs:Pong',
)
register(
    id='ping_pong_multi-v0',
    entry_point='gym_multi_envs.envs:PongMultiAgent',
)
register(
    id='snake-v0',
    entry_point='gym_multi_envs.envs:SnakeGame',
)
register(
    id='snake-v2',
    entry_point='gym_multi_envs.envs:SnakeGameV2',
    kwargs={'WIDTH': 720, 'HEIGHT': 480, 'FPS': 200}
)
register(
    id='snake-window-v2',
    entry_point='gym_multi_envs.envs:SnakeGameV2',
    kwargs={'WIDTH': 2000, 'HEIGHT': 2000, 'FPS': 200}
)

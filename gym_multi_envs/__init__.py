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

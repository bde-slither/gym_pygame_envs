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
    kwargs={'WIDTH': 180, 'HEIGHT': 120, 'FOOD_COUNT': 10, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':0, 'DIE':0}
)
register(
    id='snake_coop-v3',
    entry_point='gym_multi_envs.envs:SnakeGameV2',
    kwargs={'WIDTH': 180, 'HEIGHT': 120, 'FOOD_COUNT': 10, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':-2, 'DIE':-2}
)
register(
    id='snake_comp-v4',
    entry_point='gym_multi_envs.envs:SnakeGameV2',
    kwargs={'WIDTH': 180, 'HEIGHT': 120, 'FOOD_COUNT': 10, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':2, 'DIE':-2}
)
register(
    id='snake-window-v2',
    entry_point='gym_multi_envs.envs:SnakeGameV2',
    kwargs={'WIDTH': 720, 'HEIGHT': 480, 'FOOD_COUNT': 30, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':0, 'DIE':0}
)
register(
    id='snake-window_coop-v3',
    entry_point='gym_multi_envs.envs:SnakeGameV2',
    kwargs={'WIDTH': 720, 'HEIGHT': 480, 'FOOD_COUNT': 30, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':-2, 'DIE':-2}
)
register(
    id='snake-window_comp-v4',
    entry_point='gym_multi_envs.envs:SnakeGameV2',
    kwargs={'WIDTH': 720, 'HEIGHT': 480, 'FOOD_COUNT': 30, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':2, 'DIE':-2}
)
register(
    id='snake-greedy-v2',
    entry_point='gym_multi_envs.envs:SnakeGameGreedyV2',
    #first snake is controlled by rl, second/last is greedy approach
    kwargs={'WIDTH': 180, 'HEIGHT': 120, 'FOOD_COUNT': 10, 'SNAKE_COUNT': 3, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':0, 'DIE':0}
)
register(
    id='snake-greedy-single-v2',
    entry_point='gym_multi_envs.envs:SnakeGameGreedySingleV2',
    #first snake is controlled by rl, second is greedy approach
    kwargs={'WIDTH': 180, 'HEIGHT': 120, 'FOOD_COUNT': 10, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':0, 'DIE':0}
)
register(
    id='snake-greedy-single-coop-v2',
    entry_point='gym_multi_envs.envs:SnakeGameGreedySingleV2',
    #first snake is controlled by rl, second is greedy approach
    kwargs={'WIDTH': 180, 'HEIGHT': 120, 'FOOD_COUNT': 10, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':2, 'DIE':0}
)
register(
    id='snake-greedy-single-comp-v2',
    entry_point='gym_multi_envs.envs:SnakeGameGreedySingleV2',
    #first snake is controlled by rl, second is greedy approach
    kwargs={'WIDTH': 180, 'HEIGHT': 120, 'FOOD_COUNT': 10, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':-2, 'DIE':0}
)
register(
    id='snake-greedy-single-window-v2',
    entry_point='gym_multi_envs.envs:SnakeGameGreedySingleV2',
    #first snake is controlled by rl, second is greedy approach
    kwargs={'WIDTH': 720, 'HEIGHT': 480, 'FOOD_COUNT': 10, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':0, 'DIE':0}
)
register(
    id='snake-greedy-single-window-coop-v2',
    entry_point='gym_multi_envs.envs:SnakeGameGreedySingleV2',
    #first snake is controlled by rl, second is greedy approach
    kwargs={'WIDTH': 720, 'HEIGHT': 480, 'FOOD_COUNT': 10, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':2, 'DIE':0}
)
register(
    id='snake-greedy-single-window-comp-v2',
    entry_point='gym_multi_envs.envs:SnakeGameGreedySingleV2',
    #first snake is controlled by rl, second is greedy approach
    kwargs={'WIDTH': 720, 'HEIGHT': 480, 'FOOD_COUNT': 10, 'SNAKE_COUNT': 2, 'FPS': 1000, 'MAX_SCORE': 50, 'MAX_STEP': 3000, 'KILL':-2, 'DIE':0}
)
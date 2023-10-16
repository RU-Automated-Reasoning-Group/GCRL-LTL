from gym.envs.registration import register

register(
    id='AntBase-v3',
    entry_point='gym_envs.envs:AntEnv',
    max_episode_steps=1000
)

register(
    id='AntUp-v3',
    entry_point='gym_envs.envs:AntEnvUp',
    max_episode_steps=1000
)

register(
    id='AntDown-v3',
    entry_point='gym_envs.envs:AntEnvDown',
    max_episode_steps=1000
)

register(
    id='AntLeft-v3',
    entry_point='gym_envs.envs:AntEnvLeft',
    max_episode_steps=1000
)

register(
    id='AntRight-v3',
    entry_point='gym_envs.envs:AntEnvRight',
    max_episode_steps=1000
)

register(
    id='AntBase-v1',
    entry_point='gym_envs.envs:AntV1Env',
    max_episode_steps=1000
)


register(
    id='AntUp-v1',
    entry_point='gym_envs.envs:AntV1UpEnv',
    max_episode_steps=1000
)

register(
    id='AntDown-v1',
    entry_point='gym_envs.envs:AntV1DownEnv',
    max_episode_steps=1000
)

register(
    id='AntLeft-v1',
    entry_point='gym_envs.envs:AntV1LeftEnv',
    max_episode_steps=1000
)

register(
    id='AntRight-v1',
    entry_point='gym_envs.envs:AntV1RightEnv',
    max_episode_steps=1000
)

register(
    id='CrossMazeAnt-v1',
    entry_point='gym_envs.envs:CrossMazeAntEnv',
    max_episode_steps=1000
)

register(
    id='HalfCheetahForwardEnv-v3',
    entry_point='gym_envs.envs:HalfCheetahForwardEnv',
    max_episode_steps=1000
)

register(
    id='HalfCheetahJumpEnv-v3',
    entry_point='gym_envs.envs:HalfCheetahJumpEnv',
    max_episode_steps=1000
)

register(
    id='Pusher2dLeftEnv-v1',
    entry_point='gym_envs.envs:Pusher2dLeftEnv',
    max_episode_steps=100
)

register(
    id='Pusher2dDownEnv-v1',
    entry_point='gym_envs.envs:Pusher2dDownEnv',
    max_episode_steps=100
)

register(
    id='Pusher2dEnv-v1',
    entry_point='gym_envs.envs:Pusher2dEnv',
    max_episode_steps=100
)

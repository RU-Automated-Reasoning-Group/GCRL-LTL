from gym.envs.registration import register

from envs.safety.zones_env import ZonesEnv


### Simple LTL Envs
register(
    id='Simple-LTL-Env-v0',
    entry_point='envs.gym_letters.simple_ltl_env:SimpleLTLEnvDefault'
)

### Safety Envs
register(
    id='Zones-1-v0',
    entry_point='envs.safety.zones_env:ZonesEnv1')

register(
    id='Zones-1-v1',
    entry_point='envs.safety.zones_env:ZonesEnv1Fixed')

register(
    id='Zones-5-v0',
    entry_point='envs.safety.zones_env:ZonesEnv5')

register(
    id='Zones-5-v1',
    entry_point='envs.safety.zones_env:ZonesEnv5Fixed')

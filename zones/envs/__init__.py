from gym.envs.registration import register

from envs.safety.zones_env import ZonesEnv
from envs.task_envs import ZonePrimitiveEnv, ZoneRandomGoalEnv


### Safety Envs
register(
    id='Zones-1-v0',
    entry_point='envs.safety.zones_env:ZonesEnv1')

register(
    id='Zones-1-v1',
    entry_point='envs.safety.zones_env:ZonesEnv1Fixed')

register(
    id='Zones-8-v0',
    entry_point='envs.safety.zones_env:ZonesEnv8')

register(
    id='Zones-8-v1',
    entry_point='envs.safety.zones_env:ZonesEnv8Fixed')


register(
    id='Zones-4-v0',
    entry_point='envs.safety.zones_env:ZonesEnv4')

register(
    id='Zones-4-v1',
    entry_point='envs.safety.zones_env:ZonesEnv4Fixed')

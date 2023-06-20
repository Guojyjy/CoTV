"""Empty init file to ensure documentation for multiagent envs is created."""

from flow.envs.multiagent.base import MultiEnv
from flow.envs.multiagent.ring.wave_attenuation import \
    MultiWaveAttenuationPOEnv
from flow.envs.multiagent.ring.wave_attenuation import \
    MultiAgentWaveAttenuationPOEnv
from flow.envs.multiagent.ring.accel import AdversarialAccelEnv
from flow.envs.multiagent.ring.accel import MultiAgentAccelPOEnv
from flow.envs.multiagent.ring.accel import FlowCAV
from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from flow.envs.multiagent.traffic_light_grid import PressLight
from flow.envs.multiagent.traffic_light_grid import CoTVAll, CoTV, CoTVNOCoord
from flow.envs.multiagent.highway import MultiAgentHighwayPOEnv
from flow.envs.multiagent.merge import MultiAgentMergePOEnv
from flow.envs.multiagent.i210 import I210MultiEnv
from flow.envs.multiagent.sumo_template import CoTVCustomEnv, CoTVAllCustomEnv, CoTVMixedCustomEnv, \
    PressLightCustomEnv, FlowCAVCustomEnv, CoTVNOCoorCustomEnv

__all__ = [
    'MultiEnv',
    'AdversarialAccelEnv',
    'MultiWaveAttenuationPOEnv',
    'MultiTrafficLightGridPOEnv',
    'MultiAgentHighwayPOEnv',
    'MultiAgentAccelPOEnv',
    'MultiAgentWaveAttenuationPOEnv',
    'MultiAgentMergePOEnv',
    'I210MultiEnv',
    # added below
    'PressLight',
    'FlowCAV',
    'CoTVAll',
    'CoTV',
    'CoTVNOCoord',
    # for SUMO template
    'PressLightCustomEnv',
    'FlowCAVCustomEnv',
    'CoTVAllCustomEnv',
    'CoTVCustomEnv',
    'CoTVMixedCustomEnv',
    'CoTVNOCoorCustomEnv'
]

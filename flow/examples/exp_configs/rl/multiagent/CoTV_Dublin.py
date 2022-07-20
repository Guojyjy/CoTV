"""CoTV under Dublin scenario (sumo template)"""
import os

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, NetParams
from flow.core.params import VehicleParams
from flow.envs.multiagent import CoTVMixedCustomEnv  # CoTVCustomEnv / CoTVAllCustomEnv
from flow.networks import SumoNetwork
from flow.utils.registry import make_create_env

# Experiment parameters
N_ROLLOUTS = 18  # number of rollouts per training iteration
N_CPUS = 18  # number of parallel workers
HORIZON = 720  # time horizon of a single rollout

SPEED_LIMIT = 15
MAX_ACCEL = 3  # maximum acceleration for autonomous vehicles, in m/s^2
MAX_DECEL = 3  # maximum deceleration for autonomous vehicles, in m/s^2

ABS_DIR = os.path.abspath(os.path.dirname(__file__)).split('flow')[0]

vehicles = VehicleParams()

flow_params = dict(
    exp_tag='Dublin_CoTV_PRate50',

    env_name=CoTVMixedCustomEnv,

    network=SumoNetwork,

    simulator='traci',

    sim=SumoParams(
        render=False,
        sim_step=1,
        restart_instance=True,
        emission_path="{}output/Dublin_CoTV_PRate50".format(ABS_DIR)
    ),

    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": SPEED_LIMIT,
            "switch_time": 3.0,
            "num_observed": 1,
            "max_accel": MAX_ACCEL,
            "max_decel": MAX_ACCEL,
            "safety_device": True,  # 'True' needs emission path to save output file
            "cav_penetration_rate": 0.5,  # used for CoTVMixedCustomEnv
            "total_veh": 275
        },
    ),

    net=NetParams(
        template={
            "net": "{}/scenarios/CoTV/Dublin/dublin.net.xml".format(ABS_DIR),
            "rou": "{}/scenarios/CoTV/Dublin/dublin_clip_rl.rou.xml".format(ABS_DIR),
            "vtype": "{}/scenarios/CoTV/Dublin/rl_vtypes.add.xml".format(ABS_DIR)}
    ),

    veh=vehicles
)

create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space_tl = test_env.observation_space_tl
act_space_tl = test_env.action_space_tl
obs_space_av = test_env.observation_space_av
act_space_av = test_env.action_space_av

# Setup PG with a single policy graph for all agents
POLICY_GRAPHS = {'cav': (PPOTFPolicy, obs_space_av, act_space_av, {}),
                 'tl': (PPOTFPolicy, obs_space_tl, act_space_tl, {})}


def policy_mapping_fn(agent_id):
    """Map a policy in RLlib."""
    if "." in agent_id:
        return "cav"
    else:
        return "tl"


policies_to_train = ['cav', 'tl']

"""Multiagent TL + CAV example (single shared policy for each type of agent)."""
import os

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs.multiagent import CoTV
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import GridRouter
from flow.controllers import RLController
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env

# Experiment parameters
N_ROLLOUTS = 18  # number of rollouts per training iteration
N_CPUS = 18  # number of parallel workers
HORIZON = 720  # time horizon of a single rollout

# Road network parameters
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 300  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1
N_ROWS = 1  # number of row of bidirectional lanes
N_COLUMNS = 1  # number of columns of bidirectional lanes
# limit vehicle driving parameters
TARGET_VELOCITY = 15  # desired velocity for all vehicles in the network, in m/s
MAX_ACCEL = 3  # maximum acceleration for autonomous vehicles, in m/s^2
MAX_DECEL = 3  # maximum deceleration for autonomous vehicles, in m/s^2

# Vehicle parameters
vehicles = VehicleParams()
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(GridRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="right_of_way",
        accel=MAX_ACCEL,
        decel=MAX_DECEL,
        max_speed=TARGET_VELOCITY,
    ))

# inflows of vehicles are place on all outer edges (listed here)
left_edges = ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
right_edges = ["right0_{}".format(i) for i in range(N_COLUMNS)]
bot_edges = ["bot{}_0".format(i) for i in range(N_ROWS)]
top_edges = ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

inflow = InFlows()
for edge in right_edges:  # S->N
    inflow.add(
        veh_type="rl",
        edge=edge,
        vehs_per_hour=120,
        begin=1,
        end=300,
        depart_lane="free",
        depart_speed="random")
for edge in bot_edges:  # W->E
    inflow.add(
        veh_type="rl",
        edge=edge,
        vehs_per_hour=240,
        begin=1,
        end=300,
        depart_lane="free",
        depart_speed="random")
for edge in left_edges:  # N->S
    inflow.add(
        veh_type="rl",
        edge=edge,
        vehs_per_hour=288,
        begin=45,
        end=345,
        depart_lane="free",
        depart_speed="random")
for edge in top_edges:  # E->W
    inflow.add(
        veh_type="rl",
        edge=edge,
        vehs_per_hour=192,
        begin=60,
        end=360,
        depart_lane="free",
        depart_speed="random")

# Integrate parameters for this module
flow_params = dict(
    # name of the experiment
    exp_tag="CoTV_1x1grid",

    # name of the flow environment the experiment is running on
    env_name=CoTV,

    # name of the network class the experiment is running on
    network=TrafficLightGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=1,
        render=False,
        emission_path="{}output/CoTV_1x1grid".format(os.getcwd().split('flow')[0]),
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": TARGET_VELOCITY,
            "switch_time": 3,
            "num_observed": 1,
            "discrete": True,
            "tl_type": "controlled",
            "num_local_edges": 4,
            "num_local_lights": 4,
            "max_accel": MAX_ACCEL,
            "max_decel": MAX_DECEL,
            "safety_device": True,  # 'True' needs emission path to save output file
            "cav_penetration_rate": 1,  # used for mixed-autonomy
            "total_veh": 70,  # 240 for 1x6 grid, 70 for 1x1 grid
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params={
            "speed_limit": TARGET_VELOCITY,
            "grid_array": {
                "short_length": SHORT_LENGTH,
                "inner_length": INNER_LENGTH,
                "long_length": LONG_LENGTH,
                "row_num": N_ROWS,
                "col_num": N_COLUMNS,
                "cars_left": N_LEFT,
                "cars_right": N_RIGHT,
                "cars_top": N_TOP,
                "cars_bot": N_BOTTOM,
            },
            "horizontal_lanes": 1,
            "vertical_lanes": 1,
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization
    # or reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='custom',
        shuffle=True,
    ),
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
    if agent_id.startswith("center"):
        return "tl"
    else:
        return "cav"


policies_to_train = ['cav', 'tl']

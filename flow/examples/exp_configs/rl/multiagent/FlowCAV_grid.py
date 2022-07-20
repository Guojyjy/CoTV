"""FlowCAV under grid maps generated in FLOW"""
import os

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from flow.envs.multiagent import FlowCAV
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import SumoCarFollowingParams, VehicleParams
from flow.core.params import InFlows
from flow.controllers import GridRouter, SimCarFollowingController
from flow.core.params import TrafficLightParams

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
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(max_speed=TARGET_VELOCITY, decel=7.5, tau=1),
    routing_controller=(GridRouter, {}))

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

# Traffic light parameters
tl_logic = TrafficLightParams(baseline=False)
phases = [{
                "duration": "40",
                "state": "GrGr"
            }, {
                "duration": "3",
                "state": "yryr"
            }, {
                "duration": "40",
                "state": "rGrG"
            }, {
                "duration": "3",
                "state": "ryry"
            }]
num_tl = N_ROWS * N_COLUMNS
for i in range(0, num_tl):
    tl_id = "center"+str(i)
    tl_logic.add(tl_id, phases=phases, programID="1", tls_type="static")

# Integrate parameters for this module
flow_params = dict(
    # name of the experiment
    exp_tag="FlowCAV_1x1grid",

    # name of the flow environment the experiment is running on
    env_name=FlowCAV,

    # name of the network class the experiment is running on
    network=TrafficLightGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=1,
        render=False,
        emission_path="{}output/FlowCAV_1x1grid".format(os.path.abspath(os.path.dirname(__file__)).split('flow')[0]),
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": TARGET_VELOCITY,
            "switch_time": 3,
            "num_observed": 0,  # super() for TrafficLightGridPOEnv
            "num_controlled": 1,
            "discrete": False,
            "tl_type": "static",
            'max_accel': MAX_ACCEL,
            'max_decel': MAX_ACCEL,
            'safety_device': True,   # 'True' needs emission path to save output file
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

    tls=tl_logic
)

create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return PPOTFPolicy, obs_space, act_space, {}


# Setup PG with a single policy graph for all agents
POLICY_GRAPHS = {'cav': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'cav'


policies_to_train = ['cav']

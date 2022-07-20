"""Runner script for single and multiagent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train_ppo.py EXP_CONFIG
"""
import argparse
import json
import sys
from copy import deepcopy

from flow.utils.rllib import FlowParamsEncoder
from flow.utils.registry import make_create_env


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train_ppo.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')

    # optional input parameters
    parser.add_argument(
        '--rl_trainer', type=str, default="rllib",
        help='the RL trainer to use. either rllib or Stable-Baselines')

    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=5000,
        help='How many total steps to perform learning over')
    parser.add_argument(
        '--rollout_size', type=int, default=1000,
        help='How many steps are in a training batch.')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory with checkpoint to restore training from.')

    return parser.parse_known_args(args)[0]


def setup_exps_rllib(flow_params,
                     n_cpus,
                     n_rollouts,
                     policy_graphs=None,
                     policy_mapping_fn=None,
                     policies_to_train=None):
    """Return the relevant components of an RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters (see flow/utils/registry.py)
    n_cpus : int
        number of CPUs to run the experiment over
    n_rollouts : int
        number of rollouts per training iteration
    policy_graphs : dict, optional
    policy_mapping_fn : function, optional
    policies_to_train : list of str, optional
        set in module in exp_configs/rl/multiagent
    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    from ray import tune
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    horizon = flow_params['env'].horizon

    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    config["num_workers"] = n_cpus
    config["train_batch_size"] = horizon * n_rollouts
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32, 32]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = horizon
    config["num_gpus"] = 1
    config["timesteps_per_iteration"] = horizon * n_rollouts
    config['no_done_at_end'] = True

    # save the flow params for replay, params.json file
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # multiagent configuration
    if policy_graphs is not None:
        print("policy_graphs", policy_graphs)
        config['multiagent'].update({'policies': policy_graphs})
    if policy_mapping_fn is not None:
        config['multiagent'].update(
            {'policy_mapping_fn': tune.function(policy_mapping_fn)})
    if policies_to_train is not None:
        config['multiagent'].update({'policies_to_train': policies_to_train})

    create_env, gym_name = make_create_env(params=flow_params)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


def train_rllib(submodule, flags):
    """Train policies using the PPO algorithm in RLlib."""
    import ray
    from ray.tune import run_experiments

    flow_params = submodule.flow_params
    n_cpus = submodule.N_CPUS
    n_rollouts = submodule.N_ROLLOUTS
    policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
    policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
    policies_to_train = getattr(submodule, "policies_to_train", None)

    alg_run, gym_name, config = setup_exps_rllib(
        flow_params, n_cpus, n_rollouts,
        policy_graphs, policy_mapping_fn, policies_to_train)

    ray.init(num_cpus=n_cpus + 1)  # , object_store_memory=200 * 1024 * 1024
    exp_config = {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 10,
        "checkpoint_at_end": True,
        "max_failures": 999,
        "stop": {
            "training_iteration": flags.num_steps,
        },
    }

    if flags.checkpoint_path is not None:
        exp_config['restore'] = flags.checkpoint_path
    run_experiments({flow_params["exp_tag"]: exp_config})


def main(args):
    """Perform the training operations."""
    # Parse script-level arguments (not including package arguments).
    flags = parse_args(args)

    # Import relevant information from the exp_config script.
    module = __import__(
        "exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    module_ma = __import__(
        "exp_configs.rl.multiagent", fromlist=[flags.exp_config])

    # Import the submodule containing the specified exp_config and determine
    # whether the environment is single agent or multiagent.
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
        assert flags.rl_trainer.lower() in ["rllib", "h-baselines"], \
            "Currently, multiagent experiments are only supported through "\
            "RLlib. Try running this experiment using RLlib: " \
            "'python train_ppo.py EXP_CONFIG'"
    else:
        raise ValueError("Unable to find experiment config.")

    # Perform the training operation.
    train_rllib(submodule, flags)


if __name__ == "__main__":
    main(sys.argv[1:])

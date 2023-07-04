import os.path
import random

import ray
from ray import tune
import argparse
import configparser
import sys
import logging
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.registry import get_trainer_class

from scenario.scen_retrieve import SumoScenario
from envs.env_register import register_env_gym
from policies import PolicyConfig


# ----- Customised functions (multiagent) -----

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # to customise different agent type
    if '.' in agent_id:
        return "policy_0"  # CAV
    else:
        return "policy_1"  # TL


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a DRL training for traffic control.",
        epilog="python train.py EXP_CONFIG")

    # ----required input parameters----
    parser.add_argument(
        '--exp_config', type=str, default='cotv_config.ini',
        help='Name of the experiment configuration file, as located in exp_configs.')

    # ----optional input parameters----
    parser.add_argument(
        '--log_level', type=str, default='ERROR',
        help='Level setting for logging to track running status.'
    )

    return parser.parse_known_args(args)[0]


def main(args):
    args = parse_args(args)
    logging.basicConfig(level=args.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"DRL training with the following CLI args: {args}")
    ray.init()

    # import experiment configuration
    config_file = args.exp_config
    config = configparser.ConfigParser()
    config.read(os.path.join('./exp_configs', config_file))
    if not config:
        logger.error(f"Unable to find the experiment configuration {config_file} in exp_configs")
    config.set('TRAIN_CONFIG', 'log_level', args.log_level)

    # 1. process SUMO scenario files to get info, required in make env
    scenario = SumoScenario(config['SCEN_CONFIG'])
    logger.info(f"The scenario cfg file is {scenario.cfg_file_path}.")

    # 2. register env and make env in OpenAIgym, then register_env in Ray
    this_env = __import__("envs", fromlist=[config.get('TRAIN_CONFIG', 'env')])
    if hasattr(this_env, config.get('TRAIN_CONFIG', 'env')):
        this_env = getattr(this_env, config.get('TRAIN_CONFIG', 'env'))
    this_env_register, env_name = register_env_gym(this_env, scenario, config['SUMO_CONFIG'], config['CONTROL_CONFIG'],
                                                   config['TRAIN_CONFIG'])
    register_env(env_name, this_env_register)
    this_env = this_env(scenario, config['SUMO_CONFIG'], config['CONTROL_CONFIG'], config['TRAIN_CONFIG'])

    # 3. set DRL algorithm/model
    configs_to_ray = PolicyConfig(env_name, config['ALG_CONFIG'], config['TRAIN_CONFIG'], config['MODEL_CONFIG']).policy

    # 4. multiagent setting
    policies = {}
    act_space_dict = {}
    obs_space_dict = {}
    for section in config.sections():
        if 'policySpec' in section:
            policy_class = config.get(section, 'policy_class', fallback=None)
            if policy_class:
                policy_class = get_trainer_class(policy_class)
            obs_space = getattr(this_env, config.get(section, 'observation_space'), None)
            act_space = getattr(this_env, config.get(section, 'action_space'), None)
            num_agents = config.getint(section, 'num_agents', fallback=1)
            for i in range(num_agents):
                num_policies = len(policies.keys())
                policies.update({'policy_' + str(num_policies): PolicySpec(policy_class, obs_space, act_space,
                                                                           {'agent_id': num_policies})})
                act_space_dict.update({str(num_policies): act_space})
                obs_space_dict.update({str(num_policies): obs_space})
    for _, spec in policies.items():
        spec.config.update({"act_space_dict": act_space_dict})
        spec.config.update({"obs_space_dict": obs_space_dict})

    if policies:
        configs_to_ray.update({"multiagent": {"policies": policies,
                                              "policy_mapping_fn": policy_mapping_fn,
                                              "policies_to_train": list(policies.keys())}})
    configs_to_ray.update({'disable_env_checking': True})  # to avoid checking non-override default obs_space...

    # 5. assign termination conditions, terminate when achieving one of them
    stop_conditions = {}
    for k, v in config['STOP_CONFIG'].items():
        stop_conditions.update({k: int(v)})

    tune.run(
        config.get('ALG_CONFIG', 'alg_name'),
        config=configs_to_ray,
        checkpoint_freq=config.getint('RAY_CONFIG', 'checkpoint_freq'),  # number of iterations between checkpoints
        checkpoint_at_end=config.getboolean('RAY_CONFIG', 'checkpoint_at_end'),
        max_failures=config.getint('RAY_CONFIG', 'max_failures'),  # times to recover from the latest checkpoint
        stop=stop_conditions,
        local_dir="./ray_results/" + config.get('TRAIN_CONFIG', 'exp_name', fallback=env_name),
    )

    ray.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])

import os.path

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

# ----- Shared critic method
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
from gym.spaces import Dict, Discrete, Box
from collections import OrderedDict


# ----- Customised functions (multiagent) -----

NUM_CAV_AGENT = 4  # grid map: each intersection has 4 incoming roads; overridden by the custom scenario


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # to customise different agent type
    if ':' in agent_id:
        return "policy_0"  # CAV
    else:
        return "policy_1"  # TL


# Shared Critic method
def central_critic_observer(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""
    max_cav_agents_per_inter = NUM_CAV_AGENT
    for each in kw:
        if each == "max_cav_agents_per_inter":
            max_cav_agents_per_inter = kw[each]
    new_obs = OrderedDict()
    for each in agent_obs.keys():
        # TL_{index}: a list of CAV agent around ({TL_index}:{CAV_index}); CAV: [approaching TL]
        if ':' in each:  # CAV agent
            opponent_obs_id = f"TL_{each.split(':')[0]}"
            new_obs.update({each: {
                "opponent_action": 0,  # filled in by FillInActions
                "opponent_obs": agent_obs[opponent_obs_id] if len(agent_obs[opponent_obs_id]) != 3
                else agent_obs[opponent_obs_id]['own_obs'],
                "own_obs": agent_obs[each] if len(agent_obs[each]) != 3 else agent_obs[each]['own_obs'],
            }})
        else:
            opponent_obs_ids = [f"{each.split('_')[1]}:{i}" for i in range(max_cav_agents_per_inter)]
            opponent_obs = []
            for each_opponent in opponent_obs_ids:
                if each_opponent in agent_obs.keys():
                    if len(agent_obs[each_opponent]) != 3:  # dict for 'opponent_action', 'opponent_obs', 'own_obs'
                        opponent_obs.extend(agent_obs[each_opponent])
                    else:
                        opponent_obs.extend(agent_obs[each_opponent]['own_obs'])
                else:
                    opponent_obs.extend([0] * 7)
            new_obs.update({each: {
                "opponent_action": np.array([0] * max_cav_agents_per_inter),  # filled in by FillInActions
                "opponent_obs": np.array(opponent_obs),
                "own_obs": agent_obs[each] if len(agent_obs[each]) != 3 else agent_obs[each]['own_obs'],
            }})
    return new_obs


class FillInActions(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches."""

    def on_postprocess_trajectory(
            self,
            worker,
            episode,
            agent_id,
            policy_id,
            policies,
            postprocessed_batch,
            original_batches,
            **kwargs
    ):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        if ':' in agent_id:  # CAV
            # find the interval of this CAV agent
            _, this_batch = original_batches[agent_id]
            max_length = 0
            interval_info_index = 0
            for i in range(len(this_batch[SampleBatch.INFOS])):
                length_sum = 0
                for each in this_batch[SampleBatch.INFOS][i].values():
                    length_sum += len(each)
                if length_sum > max_length:
                    interval_info_index = i
                    max_length = length_sum
            interval_info = this_batch[SampleBatch.INFOS][interval_info_index]
            other_id = f"TL_{agent_id.split(':')[0]}"
            action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(2))
            # set the opponent actions into the observation
            _, opponent_batch = original_batches[other_id]
            opponent_actions = np.array(
                [action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]]
            )
            # extract TL action during the CAV agent interval
            opponent_actions_clip = []
            time_lasting = 0
            for duration_part in interval_info.values():
                if len(duration_part) == 1:  # vehicle not finish trip til the simulation ends
                    duration_part.append(720)
                time_lasting += (duration_part[1] - duration_part[0])
                opponent_actions_clip.extend(opponent_actions[duration_part[0]:duration_part[1]])
            if time_lasting > len(to_update):
                opponent_actions_clip = opponent_actions_clip[:-(time_lasting - len(to_update))]
            elif time_lasting < len(to_update):
                if interval_info:
                    opponent_actions_clip.extend(opponent_actions[list(interval_info.values())[-1][1]:
                                                                  list(interval_info.values())[-1][1]
                                                                  + (len(to_update) - time_lasting)])
                else:
                    opponent_actions_clip.extend(opponent_actions[:(len(to_update) - time_lasting)])
            if len(opponent_actions_clip) < len(to_update):
                diff = len(to_update) - len(opponent_actions_clip)
                opponent_actions_clip.extend(opponent_actions[:diff])
            to_update[:, :2] = np.array(opponent_actions_clip, dtype=object)
        else:  # TL
            default_to_update = np.array([[0] * NUM_CAV_AGENT] * 720)
            other_id = [f"{agent_id.split('_')[1]}:{i}" for i in range(NUM_CAV_AGENT)]
            # set the opponent actions into the observation
            _, this_batch = original_batches[agent_id]
            interval_info = {}
            for i in range(len(this_batch[SampleBatch.INFOS])):
                for each_opponent in this_batch[SampleBatch.INFOS][i].keys():
                    if each_opponent not in interval_info.keys():
                        interval_info.update({each_opponent: this_batch[SampleBatch.INFOS][i][each_opponent]})
            for each in other_id:
                if each in original_batches.keys():
                    _, opponent_batch = original_batches[each]
                    opponent_actions = np.array(opponent_batch[SampleBatch.ACTIONS])
                    time_lasting = 0
                    for duration_part in interval_info[each].values():
                        duration_diff = duration_part[1] - duration_part[0]
                        opponent_actions_part_length = len(opponent_actions[time_lasting:
                                                           time_lasting + duration_diff])
                        if opponent_actions_part_length != duration_diff:
                            default_to_update[time_lasting:time_lasting + opponent_actions_part_length,
                                              int(each.split(":")[1]):int(each.split(":")[1]) + 1] = \
                                np.array(opponent_actions[time_lasting:time_lasting + opponent_actions_part_length],
                                         dtype=object).reshape(opponent_actions_part_length, 1)
                        else:
                            default_to_update[time_lasting:time_lasting + duration_diff,
                                              int(each.split(":")[1]):int(each.split(":")[1]) + 1] = \
                                np.array(opponent_actions[time_lasting:time_lasting + duration_diff],
                                         dtype=object).reshape(duration_diff, 1)
                        time_lasting += duration_diff
            to_update[:, :NUM_CAV_AGENT] = default_to_update


# ------


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a DRL training for traffic control.",
        epilog="python train_centralized2.py EXP_CONFIG")

    # ----required input parameters----
    parser.add_argument(
        '--exp_config', type=str, default='mcotv_config.ini',
        help='Name of the experiment configuration file, as located in exp_configs.')

    # ----optional input parameters----
    parser.add_argument(
        '--log_level', type=str, default='ERROR',
        help='Level setting for logging to track running status.'
    )

    parser.add_argument(
        '--framework', choices=["tf", "tf2"], default='tf',
        help='DL framework specifier.'
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
    config.set('TRAIN_CONFIG', 'framework', args.framework)

    # 1. process SUMO scenario files to get info, required in make env
    scenario = SumoScenario(config['SCEN_CONFIG'])
    _, max_cav_agents_per_inter, _, _ = scenario.node_mapping()
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
    policy_0_observer_space = Dict({
        "own_obs": Box(low=-5, high=5, shape=(7,)),
        "opponent_obs": Box(low=0., high=1, shape=(25,)),  # 49 for Dublin
        "opponent_action": Discrete(2)
    })  # CAV
    policy_1_observer_space = Dict({
        "own_obs": Box(low=0., high=1, shape=(25,)),
        "opponent_obs": Box(low=-5, high=5, shape=(7 * max_cav_agents_per_inter,)),
        "opponent_action": Box(low=-3, high=3, shape=(max_cav_agents_per_inter,)),
    })  # TL
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
                if num_policies == 0:
                    policies.update({'policy_' + str(num_policies): PolicySpec(policy_class, policy_0_observer_space,
                                                                               act_space, {})})
                else:
                    policies.update({'policy_' + str(num_policies): PolicySpec(policy_class, policy_1_observer_space,
                                                                               act_space, {})})

    if policies:
        configs_to_ray.update({"multiagent": {"policies": policies,
                                              "policy_mapping_fn": policy_mapping_fn,
                                              "policies_to_train": list(policies.keys()),
                                              "observation_fn": central_critic_observer}})
    configs_to_ray.update({"callbacks": FillInActions})
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

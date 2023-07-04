"""Register a custom env with a string name,
create and return an env instance
"""

import gym
from gym.envs.registration import register


def register_env_gym(env, scenario, sumo_config, control_config, train_config, version=0):
    """
    Create a rllib custom environment compatible with OpenAI gym

    Parameters
    ---------
    scenario
    train_config
    control_config
    sumo_config
    env
    version: int, optional
        environment version number, required by gym env name

    Returns
    ------
    gym.envs.make(env):
        method that calls OpenAI gym's register method and make method
    env_name:
        name of the created gym environment, i.e., str of the env class
    """

    # exp_name = params["exp_name"]
    env_name = env.__name__

    # deal with multiple environments being created under the same name
    all_envs = gym.envs.registry.all()
    env_ids = [each_env.id for each_env in all_envs]
    while "{}-v{}".format(env_name, version) in env_ids:
        version += 1
    env_name = "{}-v{}".format(env_name, version)

    def create_env(*_):

        parameter = {
            "scenario": scenario,
            "sumo_config": sumo_config,
            "control_config": control_config,
            "train_config": train_config,
        }

        register(
            id=env_name,  # require ^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$
            entry_point=env.__module__ + ':' + env.__name__,
            order_enforce=False,  # ERROR: rollout_worker.py line640+ not a subclass
            kwargs=parameter)

        return gym.envs.make(env_name)

    return create_env, env_name


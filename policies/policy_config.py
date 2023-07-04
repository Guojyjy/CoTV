from ray.rllib.models import ModelCatalog
from policies.models.CentralizedCriticModel import CentralizedCriticModel
import ast


def ppo_config(train_configs, env_name, model_configs):

    model = {}
    for key in model_configs.keys():
        if key == 'name':
            model.update({'custom_model': model_configs.get(key)})
            ModelCatalog.register_custom_model(
                model_configs.get('name'),
                CentralizedCriticModel,
            )
        else:
            model.update({key: ast.literal_eval(model_configs.get(key))})

    return {
        "env": env_name,
        "log_level": train_configs.get('log_level'),
        "num_workers": train_configs.getint('num_workers'),
        "train_batch_size": train_configs.getint('horizon') * train_configs.getint('num_workers')
        if not train_configs.getint('train_batch_size') else train_configs.getint('train_batch_size'),
        "gamma": 0.999,   # discount rate
        "model": model,
        "use_gae": True,
        "lambda": 0.97,
        "kl_target": 0.02,
        "num_sgd_iter": 10,
        "horizon": train_configs.getint('horizon'),
        "timesteps_per_iteration": train_configs.getint('horizon') * train_configs.getint('num_workers'),
        "no_done_at_end": True,
    }


def maddpg_config(train_configs, env_name, model_configs):

    return {
        "env": env_name,
        "log_level": train_configs.get('log_level'),
        "num_workers": train_configs.getint('num_workers'),
        "train_batch_size": train_configs.getint('train_batch_size'),
        "horizon": train_configs.getint('horizon'),
        "rollout_fragment_length": 720,
        "gamma": 0.95,  # discount rate

    }


class PolicyConfig:

    def __init__(self, env_name, alg_configs, train_configs, model_configs=None):
        self.env_name = env_name
        self.name = alg_configs.get('alg_name')
        self.policy = self.find_policy(train_configs, model_configs)

    def find_policy(self, train_configs, model_configs):
        if self.name == 'PPO':
            return ppo_config(train_configs, self.env_name, model_configs)
        if self.name == 'contrib/MADDPG':
            return maddpg_config(train_configs, self.env_name, model_configs)




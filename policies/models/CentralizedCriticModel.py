from gym.spaces import Box

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


class CentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        if obs_space.shape == (34,):  # 58 for dublin, 34 - CAV extended obs
            self.action_model = FullyConnectedNetwork(
                Box(low=-5, high=5, shape=(7,)),  # CAV obs fixed
                action_space,
                num_outputs,
                model_config,
                name + "_action",
            )
        else:
            self.action_model = FullyConnectedNetwork(
                Box(low=0., high=1, shape=(25,)),
                action_space,  # 49 for dublin, 25, TL obs
                num_outputs,
                model_config,
                name + "_action",
            )

        self.value_model = FullyConnectedNetwork(
            obs_space, action_space, 1, model_config, name + "_vf"
        )

    def forward(self, input_dict, state, seq_lens):
        self._value_out, _ = self.value_model(
            {"obs": input_dict["obs_flat"]}, state, seq_lens
        )
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

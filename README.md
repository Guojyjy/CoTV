# CoTV, cooperative control for traffic light signals and CAV using DRL

The experiments are conducted on a simulator platform [SUMO](https://www.eclipse.org/sumo/). 
[RLlib](https://docs.ray.io/en/latest/rllib.html#) is an open-source library for reinforcement learning.

**CoTV** and **M-CoTV** are implemented on a custom framework, called `Coach`.

## Local installation

### Install Anaconda

It is highly recommended to install [Anaconda](https://www.anaconda.com/products/individual) that is convenient to set up a specific environment for Flow and its dependencies.

### Install Coach

- Create a conda environment for Coach: `conda create -n coach python=3.8`
- Switch to the environment: `conda activate coach`
- Deploy RLlib with a deep-learning framework, such as TensorFlow: `pip install "ray[rllib]" tensorflow`
- Other dependencies: `pip install lxml`
- More details: `environment.yaml`

### Install SUMO

It is highly recommended to use the installation methods from [Downloads-SUMO documentation](https://sumo.dlr.de/docs/Downloads.php). 

```shell
# run the following commands to check the version/location information or load SUMO GUI
which sumo
sumo --version
sumo-gui
```

Experiments can be performed based on:
    
| OS    | SUMO   | TensorFlow | Ray    |
|-------|--------|------------|--------|
| MacOS | 1.12.0 | 2.4.1      | 1.11.0 |
| Linux | 1.10.0 | 2.9.1      | 1.13.0 |

## Experiments

Enter the project in the specific environment:

```shell
cd ~/CoTV
conda activate coach
```

### Traffic control methods

#### CoTV 

```shell
python train.py cotv_config.ini
python train_centralized.py mcotv_config.ini 
```

- ***exp_configs*** to define exp configurations, including
  - algorithm [ALG_CONFIG]
  - model [MODEL_CONFIG]
  - scenario [SCEN_CONFIG]
  - sumo simulation [SUMO_CONFIG]
  - training and termination conditions [TRAIN_CONFIG], [STOP_CONFIG]
  - ray [RAY_CONFIG]
  - agent policy and model control, e.g., [CAV_policySpec], [TL_policySpec], and [CONTROL_CONFIG]
- System design in ***envs/***
  - ***CoTVEnv*** is the proposed model
  - ***MCoTVEnv*** is the implementation of CoTV with agent cooperation in action and state
    - The custom model is _**policies/models/CentralizedCriticModel.py**_
    - Due to the different sizes of TL state space (i.e., varying numbers of incoming roads, especially for urban scenarios), some settings are hard-coded for grid maps or urban networks:
    ```python
    # From line 226 in train_centralized.py
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
    ```
    ```python
    # From line 27 in policies/models/CentralizedCriticModel.py
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
    ```

### Output files

Experiment output files set in **_/output_**, according to the corresponding **exp_config** file: _output_path_ in [SUMO_CONFIG]

### Evaluation

-  _**/evaluation/outputFilesProcessing.py**_ filters the output files in **_CoTV/output_**
  - delete outdated, incomplete, and initial (size < 4kB ) files
  - recover the format of xml files

- ***/evaluation/getResults.py*** gets traffic statistic
  - Travel time
  - Delay
  - Fuel consumption
  - CO<sub>2</sub> emissions
  - TTC, total number of possible rear-end collisions


-----

#### Citing

```latex
@article{guo2023cotv,
  title={CoTV: Cooperative Control for Traffic Light Signals and Connected Autonomous Vehicles Using Deep Reinforcement Learning},
  author={Guo, Jiaying and Cheng, Long and Wang, Shen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023},
  publisher={IEEE}
}
```

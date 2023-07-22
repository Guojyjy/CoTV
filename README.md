# CoTV, cooperative control for traffic light signals and CAV using DRL

[https://github.com/Guojyjy/CoTV/blob/main/CoTV%20demo.mp4](https://github.com/Guojyjy/CoTV/assets/109638662/92b6d8f5-c1ed-4edb-a158-6dc2f4646043)

The experiments are conducted on a simulator platform [SUMO](https://www.eclipse.org/sumo/). The model design and implementation are based on [Flow](https://flow-project.github.io). [RLlib](https://docs.ray.io/en/latest/rllib.html#) is an open-source library for reinforcement learning.

## Local installation

### Install Anaconda

It is highly recommended to install [Anaconda](https://www.anaconda.com/products/individual) that is convenient to set up a specific environment for Flow and its dependencies.

### Install FLOW

Please download the project. It covers the whole framework of [Flow](https://github.com/flow-project/flow) and my model implementation based on Flow.

```shell
git clone git@github.com:Guojyjy/CoTV.git
```

#### Create a conda environment

Running the related scripts to create the environment, install Flow and its dependencies requires `cd ~/CoTV/flow`, then enter the below commands:

```shell
conda env create -f environment.yml
conda activate flow
python setup.py develop # if the conda install fails, try the next command to install the requirements using pip
pip install -e . # install flow within the environment
```

The Flow documentation provides more installation details: [Local installation of Flow](https://flow.readthedocs.io/en/latest/flow_setup.html#).

Please note that the definition of `$SUMO_HOME` within the installing process of SUMO would cause an error in the installation of Flow so that please install Flow first.

### Install SUMO

It is highly recommended to use the installation methods from [Downloads-SUMO documentation](https://sumo.dlr.de/docs/Downloads.php). 

The experiments shown in the paper were conducted on SUMO Version 1.10.0.

The instructions covered in [Installing Flow and SUMO](https://flow.readthedocs.io/en/latest/flow_setup.html#installing-flow-and-sumo) from Flow documentation is outdated.

```shell
# run the following commands to check the version/location information or load SUMO GUI
which sumo
sumo --version
sumo-gui
```

#### Troubleshooting

1. See output like `Warning: Cannot find local schema '../sumo/data/xsd/types_file.xsd', will try website lookup.`
   - Set `$SUMO_HOME` to `$../sumo `instead of `$../sumo/bin`
2. Error like `ModuleNotFoundError: No module named 'flow'`, `ImportError: No module named flow.subpackage`
   - `pip install -e . ` to  install flow within the environment, mentioned at **Install FLOW**
   - Issue on inconsistent version of python required in the environment
     - Linux version of SUMO contains python in `/sumo/bin/` which may cause the error.
     - `which python` to check the current used
     - `echo $PATH` to check the order of the directories in the path variable to look for python
     - Add `export PATH=/../anaconda3/env/flow/bin:$PATH` in the file `~/.bashrc`
     - Restart the terminal,  update the configuration through `source ~/.bashrc`

## Virtual installation 

We have built a docker image to simplify the installation of project. 

To run a docker container based on the CoTV docker image:

```shell
# first pull the image from docker hub and run a container
# -d, run container in background and print container ID
# --env, --volume, allow to execute sumo-gui
docker run -dit --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" gjyjy/cotv:latest
# find the container id
docker ps
# interact with the running container in terminal
docker exec -it CONTAINER_ID bash
# exit the container
exit / [ctrl+D]
# stop the container
docker container stop CONTAINER_ID
```

If having an error about `FXApp::openDisplay: unable to open display` when using sumo-gui, the permission of X server host should be adjusted on your local machine:

``` shell
xhost +local:
```

## Experiments

Enter the project in the specific environment:

```shell
cd ~/CoTV/flow
conda activate flow
```

### Traffic control methods

#### CoTV 

```shell
# for 1x1 and 1x6 grid maps
python examples/train_ppo.py CoTV_grid --num_steps 150
# for Dublin scenario, i.e., six consecutive intersections, or another extended Dublin scenario covering almost 1km^2
python examples/train_ppo.py CoTV_Dublin 
```

- ***train_ppo.py*** includes DRL algorithm assigned and parameter settings
  - *num_gpu*, *num_worker* specify the computation resource used in the DRL training
- **CoTV_grid** and **CoTV_Dublin** both correspond to the modules of  ***flow/examples/exp_configs/rl/multiagent***  including all setting of road network and DRL agent 
- *"**--num_steps**"* is the termination condition for DRL training, optional
- The SUMO files of Dublin scenario locate in ***CoTV/scenarios***
- System design in ***flow/flow/env/multiagent***
  - ***traffic_light_grid.py*** for grid maps
    - ***CoTV*** is action-independent multi-agent DRL model with cooperation schemes in the state exchange, for full-autonomy and mixed-autonomy traffic.
    - ***CoTVAll*** is the implementation of *CoTV* *, controlling all CAVs under full-autonomy traffic.
    - ***CoTVNOCoord*** is the implementation of *I-CoTV*, combining independent policy training on the two type of agents (CAV and traffic light controller). There is no cooperation design in either state or action.
  - ***sumo_template.py*** for Dublin scenarios
    - ***CoTVCustomEnv*** for full-autonomy traffic.
    - ***CoTVMixedCustomEnv*** for mixed-autonomy traffic.
    - ***CoTVAllCustomEnv*** is the implementation of *CoTV* *.
    - ***CoTVNOCoorCustomEnv*** is the implementation of *I-CoTV*.

NOTE: CoTV and M-CoTV are implemented and uploaded in another branch [M-CoTV](https://github.com/Guojyjy/CoTV/tree/M-CoTV). My customized DRL framework, named Coach, supports traffic control under various simulated road scenarios provided by SUMO, meanwhile, simplifying the experiment configuration required on Flow. CoTV can achieve the same level of traffic improvements as running on Flow.


#### PressLight[1]

```shell
python examples/train_dqn.py PressLight_grid
python examples/train_dqn.py PressLight_Dublin
```

#### FixedTime

```shell
python examples/train_dqn.py FixedTime_grid
python examples/train_dqn.py FixedTime_Dublin
```

Implement based on PressLight with specific setting in the modules of  ***flow/examples/exp_configs/rl/multiagent***

- Static traffic light

  ```python
  env=EnvParams(
          additional_params={
              "static": True
          },
      )
  ```

#### GLOSA

```shell
python examples/train_dqn.py GLOSA_grid
python examples/train_dqn.py GLOSA_Dublin
```

Implement based on PressLight with specific setting in the modules of  ***flow/examples/exp_configs/rl/multiagent***

- Equip the GLOSA device for all vehicles, see [SUMO/GLOSA](https://sumo.dlr.de/docs/Simulation/GLOSA.html)
- Actuated traffic light, see [SUMO/Traffic Lights that respond to traffic/Type 'actuated'](https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#type_actuated)

#### FlowCAV[2]

```shell
python examples/train_ppo.py FlowCAV_grid
python examples/train_ppo.py FlowCAV_Dublin
```

### Output files

- Road network configuration files for SUMO during the training process in **_flow/flow/core/kernel/network/debug_**

- Experiment output files set in **_CoTV/output_**, according to the emission path in the modules of **_flow/examples/exp_configs/rl_**

### Evaluation

-  _**CoTV/evaluation/outputFilesProcessing.py**_ filters the output files in **_CoTV/output_**
  - delete outdated, incomplete, and initial (size < 4kB ) files
  - recover the format of xml files

- ***CoTV/evaluation/getResults.py*** gets traffic statistic
  - Travel time
  - Delay
  - Fuel consumption
  - CO<sub>2</sub> emissions
  - TTC, total number of possible rear-end collisions


-----

<img width="470" alt="CoTV poster" src="https://github.com/Guojyjy/CoTV/assets/109638662/90b9ed49-9907-4048-8b4d-a2f4708b9dc8">

[CoTV poster.pdf](https://github.com/Guojyjy/CoTV/blob/main/CoTV%20poster.pdf)

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

#### References

[1] _Wei, Hua, et al. "Presslight: Learning max pressure control to coordinate traffic signals in arterial network." *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. 2019._

[2] _Wu, Cathy, et al. "Flow: Architecture and benchmarking for reinforcement learning in traffic control." *arXiv preprint arXiv:1710.05465* 10 (2017)._


from envs.env_cotv import CoTVEnv
import numpy as np
from gym.spaces import Box, Discrete


class MCoTVEnv(CoTVEnv):

    def __init__(self, scenario, sumo_config, control_config, train_config):
        super().__init__(scenario, sumo_config, control_config, train_config)

        # --- for shared centralized method ---
        tls_id = list(self.states_tl.keys())
        self.TL_ids_mapping = {tls_id[i]: f"TL_{i}" for i in range(len(tls_id))}
        self.CAV_ids_mapping = {}
        # record the duration of each CAV agent
        # {cav_agent_mapping: {cav_id: [start_simulationStep, end_simulationStep]}}
        self.CAV_duration = {}

    def _get_state(self):
        obs = {}

        if not self.max_speed:
            self.max_speed = self.scenario.max_speed()
            # vehicles will have a random speedFactor with a deviation of 0.1 and mean of 1.0
            # which means there will be different desired speeds in the vehicle population by default
            self.max_length = self.scenario.max_length()
        accel_norm = self.max_accel + self.max_decel_norm

        veh_ids_lane = {each: [] for each in self.veh_list_lane.keys()}
        for each_veh in self.sumo.vehicle.getIDList():
            lane_id = self.sumo.vehicle.getLaneID(each_veh)
            if lane_id in self.veh_list_lane.keys():  # ignore internal links of road network
                veh_ids_lane[self.sumo.vehicle.getLaneID(each_veh)].append(each_veh)

        max_num_veh_c = self.max_length / 7.5  # 7.5 = vehicle len + min gap
        veh_num_per_lane = {}  # {lane_id: num_veh}
        for each in self.veh_list_lane.keys():
            now_veh_id_list = veh_ids_lane[each]
            pre_veh_id_list = list(self.veh_list_lane[each].keys())
            for each_veh in pre_veh_id_list:
                if each_veh not in now_veh_id_list:
                    del self.veh_list_lane[each][each_veh]
                    # add the end timestep for each vehicle as the CAV agent
                    if each_veh in self.CAV_ids_mapping.keys():
                        cav_id_mapping = self.CAV_ids_mapping[each_veh]
                        if len(self.CAV_duration[cav_id_mapping][each_veh]) < 2:
                            self.CAV_duration[cav_id_mapping][each_veh].append(self.step_count_in_episode)
            # update vehicles (add vehicles newly departed)
            for veh in now_veh_id_list:
                self.veh_list_lane[each].update({veh: self.update_vehicles(veh, each, accel_norm)})
            veh_num_per_lane.update({each: len(now_veh_id_list) / max_num_veh_c})

        # Observed vehicle information for traffic light controller
        speeds = []
        accels = []
        dist_to_junc = []
        lane_index = []
        all_observed_ids = []
        # traffic light information
        for tl_id, lanes in self.mapping_inc.items():
            local_speeds = []
            local_accels = []
            local_dist_to_junc = []
            local_lane_index = []
            for lane in lanes:
                # sort to select the closest vehicle
                veh_id_sort = {}
                for veh in self.veh_list_lane[lane].keys():
                    veh_id_sort.update({self.veh_list_lane[lane][veh][2]: veh})
                num_observed = min(self.num_observed, len(self.veh_list_lane[lane]))
                observed_ids = [veh_id_sort[sorted(veh_id_sort.keys())[i]] for i in range(num_observed)]
                all_observed_ids.extend(observed_ids)

                local_speeds.extend([self.veh_list_lane[lane][veh_id][0] for veh_id in observed_ids])
                local_accels.extend([self.veh_list_lane[lane][veh_id][1] for veh_id in observed_ids])
                local_dist_to_junc.extend([self.veh_list_lane[lane][veh_id][2] for veh_id in observed_ids])
                local_lane_index.extend([self.veh_list_lane[lane][veh_id][3] for veh_id in observed_ids])

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    local_speeds.extend([1] * diff)
                    local_accels.extend([0] * diff)
                    local_dist_to_junc.extend([1] * diff)
                    local_lane_index.extend([0] * diff)

            # not 4-leg intersection
            if len(lanes) < self.num_int_lane_max:
                diff = self.num_int_lane_max - len(lanes)
                local_speeds.extend([1] * diff * self.num_observed)
                local_accels.extend([0] * diff * self.num_observed)
                local_dist_to_junc.extend([1] * diff * self.num_observed)
                local_lane_index.extend([0] * diff * self.num_observed)

            speeds.append(local_speeds)
            accels.append(local_accels)
            dist_to_junc.append(local_dist_to_junc)
            lane_index.append(local_lane_index)
        self.observed_ids = all_observed_ids

        # add observation of TL
        for tl_id in self.mapping_inc.keys():
            tl_id_num = list(self.mapping_inc.keys()).index(tl_id)
            local_inc_lanes = self.mapping_inc[tl_id]
            local_out_lanes = self.mapping_out[tl_id]

            veh_num_per_in = [veh_num_per_lane[each] for each in local_inc_lanes]
            veh_num_per_out = [veh_num_per_lane[each] for each in local_out_lanes]
            # not 4-leg intersection
            if len(local_inc_lanes) < self.num_int_lane_max:
                diff = self.num_int_lane_max - len(local_inc_lanes)
                veh_num_per_in.extend([0] * diff)
            if len(local_out_lanes) < self.num_out_lane_max:
                diff = self.num_out_lane_max - len(local_out_lanes)
                veh_num_per_out.extend([0] * diff)

            states = self.states_tl[tl_id]
            now_state = self.sumo.trafficlight.getRedYellowGreenState(tl_id)
            state_index = states.index(now_state)

            observation = np.array([round(i, 8) for i in np.concatenate(
                [speeds[tl_id_num], accels[tl_id_num], dist_to_junc[tl_id_num], lane_index[tl_id_num],
                 veh_num_per_in, veh_num_per_out, [state_index / len(states)]]
            )])
            obs.update({self.TL_ids_mapping[tl_id]: observation})

        # add observation of CAV
        for cav_id in self.observed_ids:
            this_speed = self.sumo.vehicle.getSpeed(cav_id)
            this_accel = self.sumo.vehicle.getAcceleration(cav_id)
            this_accel = (this_accel, -15)[abs(this_accel) >= 15]

            if self.sumo.vehicle.getNextTLS(cav_id):
                incoming_tl = self.sumo.vehicle.getNextTLS(cav_id)[0][0]  # [(tlsID, tlsIndex, distance, state), ...]
                this_dist_to_junc = self.sumo.vehicle.getNextTLS(cav_id)[0][2]
            else:  # TODO: check whether this situation exists
                incoming_tl = None
                this_dist_to_junc = self.max_length

            if incoming_tl:
                states = self.states_tl[incoming_tl]
                now_state = self.sumo.trafficlight.getRedYellowGreenState(incoming_tl)
                state_index = states.index(now_state)
                state_norm = state_index / len(states)
            else:
                state_norm = 0

            if self.sumo.vehicle.getLeader(cav_id):
                lead_veh = self.sumo.vehicle.getLeader(cav_id)[0]
                lead_speed = self.sumo.vehicle.getSpeed(lead_veh)
                lead_gap = self.sumo.vehicle.getLeader(cav_id)[1]
                lead_accel = self.sumo.vehicle.getAcceleration(lead_veh)
                lead_accel = (lead_accel, -15)[abs(lead_accel) >= 15]
            else:
                lead_speed = self.max_speed + this_speed
                lead_gap = self.max_length
                lead_accel = 3

            if lead_gap / self.max_length > 5:
                lead_gap = 5 * self.max_length
            elif lead_gap / self.max_length < -5:
                lead_gap = -5 * self.max_length

            # --- for shared centralized method ---
            lane_id = self.sumo.vehicle.getLaneID(cav_id)
            cav_id_mapping = f"{self.TL_ids_mapping[incoming_tl].split('_')[1]}:" \
                             f"{list(self.mapping_inc[incoming_tl]).index(lane_id)}"
            self.CAV_ids_mapping.update({cav_id: cav_id_mapping})
            if cav_id_mapping not in self.CAV_duration.keys():
                # add when the CAV agent on this incoming road shows at first time
                self.CAV_duration.update({cav_id_mapping: {cav_id: [self.step_count_in_episode]}})
            else:
                if cav_id not in self.CAV_duration[cav_id_mapping].keys():
                    # replace the CAV agent with another vehicle labelled by its real id
                    self.CAV_duration[cav_id_mapping].update({cav_id: [self.step_count_in_episode]})

            obs.update({cav_id_mapping: np.array([
                this_dist_to_junc / self.max_length, this_speed / self.max_speed,
                this_accel / self.max_accel, (lead_speed - this_speed) / self.max_speed,
                lead_gap / self.max_length, lead_accel / self.max_accel,
                state_norm])})

        self.observation_info = obs
        return obs

    def _compute_reward(self):
        reward = {}
        for each_tl in self.states_tl.keys():
            obs = list(self.observation_info[self.TL_ids_mapping[each_tl]])
            num_vehicle_start_index = 4 * self.num_int_lane_max * self.num_observed
            in_traffic_sum = np.sum(obs[num_vehicle_start_index:num_vehicle_start_index + self.num_int_lane_max])
            out_traffic_sum = np.sum(obs[num_vehicle_start_index + self.num_int_lane_max:
                                         num_vehicle_start_index + self.num_int_lane_max + self.num_out_lane_max])
            reward.update({self.TL_ids_mapping[each_tl]: -(in_traffic_sum - out_traffic_sum)})

        for each_cav in self.observed_ids:
            lane_id = self.sumo.vehicle.getLaneID(each_cav)
            veh_ids = list(self.veh_list_lane[lane_id].keys())
            reward.update({self.CAV_ids_mapping[each_cav]: - self.avg_speed_diff(veh_ids, lane_id)
                                                           - self.stable_acceleration_positive(veh_ids)})

        return reward

    def _get_info(self):
        infos = {}
        for each in self.CAV_duration.keys():
            if each in self.observation_info.keys():
                infos.update({each: self.CAV_duration[each]})
        for each in self.states_tl.keys():
            infos.update({self.TL_ids_mapping[each]: {}})
            for cav_around in self.CAV_duration.keys():
                if cav_around.split(":")[0] == self.TL_ids_mapping[each].split("_")[1] and \
                        cav_around in self.observation_info.keys():
                    infos[self.TL_ids_mapping[each]].update({cav_around: infos[cav_around]})
        return infos

    def _apply_actions(self, actions):
        for agent_id, action in actions.items():
            if agent_id not in self.TL_ids_mapping.values():
                # perform acceleration actions for CAV agents
                agent_id = list(self.CAV_ids_mapping.keys())[list(self.CAV_ids_mapping.values()).index(agent_id)]
                if agent_id in self.sumo.vehicle.getIDList():
                    speed_now = self.sumo.vehicle.getSpeed(agent_id)
                    speed_next = max(speed_now + action * self.sim_step, 0)
                    self.sumo.vehicle.slowDown(agent_id, speed_next, self.sim_step)
            else:
                # perform signal switching for traffic light controller
                agent_id = list(self.TL_ids_mapping.keys())[list(self.TL_ids_mapping.values()).index(agent_id)]
                switch = action > 0
                states = self.states_tl[agent_id]
                now_state = self.sumo.trafficlight.getRedYellowGreenState(agent_id)
                state_index = states.index(now_state)
                if switch and 'G' in now_state:
                    self.sumo.trafficlight.setPhase(agent_id, state_index + 1)

"""Environment for training CAV and/or TL in a sumo net template."""
from random import choice

import numpy as np
from gym.spaces import Box
from gym.spaces.discrete import Discrete

from flow.core import rewards
from flow.envs.multiagent.base import MultiEnv

import math

ADDITIONAL_ENV_PARAMS_CAV = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 15,
}

ADDITIONAL_ENV_PARAMS_TL = {
    # minimum switch time for each traffic light (in seconds), yellow light
    "switch_time": 3.0,
}

ADDITIONAL_ENV_PARAMS_CAVTL = {
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 15,
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 3.0,
    # num of vehicles the agent can observe on each incoming edge
    "num_observed": 1,
    # # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
}


class CoTVCustomEnv(MultiEnv):
    """
        To adopt network from sumo
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_CAVTL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # network, edge
        self.mapping_inc, self.num_local_edges_max, self.mapping_out, self.num_out_edges_max = network.node_mapping
        self.lanes_related = []
        for each in self.mapping_inc.values():
            self.lanes_related.extend(each)
        for each in self.mapping_out.values():
            self.lanes_related.extend(each)
        # traffic light
        self.states_tl = network.get_states()
        self.num_traffic_lights = len(self.mapping_inc.keys())
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        # vehicle
        self.observation_info = {}
        self.leader = []
        # vehs list for the edges around intersections in the network
        self.vehs_edges = {i: {} for i in self.lanes_related}
        # used during visualization
        self.observed_ids = []
        # exp setting
        self.num_observed = env_params.additional_params.get("num_observed", 1)
        self.min_switch_time = env_params.additional_params["switch_time"]
        self.target_speed = env_params.additional_params.get("target_velocity", 15)

    @property
    def action_space_tl(self):
        return Discrete(2)

    @property
    def action_space_av(self):
        return Box(low=-abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"], shape=(1,))

    @property
    def observation_space_tl(self):
        return Box(low=0., high=1, shape=(4 * self.num_local_edges_max * self.num_observed +
                                          self.num_local_edges_max + self.num_out_edges_max + 3,))

    @property
    def observation_space_av(self):
        return Box(low=-5, high=5, shape=(9,))

    def full_name_edge_lane(self, veh_id):
        edge_id = self.k.vehicle.get_edge(veh_id)
        lane_id = self.k.vehicle.get_lane(veh_id)
        return edge_id + '_' + str(lane_id)

    def convert_edge_into_num(self, edge_id):
        return self.lanes_related.index(edge_id) + 1

    def get_observed_info_veh(self, veh_id, max_speed, max_length, accel_norm):
        speed_veh = self.k.vehicle.get_speed(veh_id) / max_speed
        accel_veh = self.k.vehicle.get_realized_accel(veh_id)
        dis_veh = (self.k.network.edge_length(self.k.vehicle.get_edge(veh_id)) -
                   self.k.vehicle.get_position(veh_id)) / max_length
        edge_veh = self.convert_edge_into_num(self.full_name_edge_lane(veh_id)) / len(self.lanes_related)
        # 0: no road to form a 4-leg intersection

        # accel normalization
        if accel_veh < -15:
            accel_veh = -15
        accel_veh = (accel_veh + 15) / accel_norm
        return [speed_veh, accel_veh, dis_veh, edge_veh]

    def get_state(self):

        obs = {}
        self.leader = []

        # normalizing constants
        max_speed = self.k.network.max_speed()
        edge_length = []
        edge_length.extend([self.k.network.edge_length(edge) for edge in self.k.network.get_edge_list()])
        max_length = max(edge_length)

        max_accel = self.env_params.additional_params["max_accel"]
        max_decel = 15  # emergence stop
        accel_norm = max_accel + max_decel

        veh_lane_pair = {each: [] for each in self.vehs_edges.keys()}
        for each_veh in self.k.vehicle.get_ids():
            # skip the internal links in intersections
            if self.full_name_edge_lane(each_veh) in self.lanes_related:
                veh_lane_pair[self.full_name_edge_lane(each_veh)].append(each_veh)
        # vehicles for incoming and outgoing - info map
        w_max = max_length / 7.5  # normalization for vehicle number, length + min gap
        veh_num_per_edge = {}  # key: name of each edge in the road network
        for each in self.vehs_edges.keys():
            all_vehs = veh_lane_pair[each]
            # remove vehicles already left the edge after one step, not restore at this step
            pre_observed_vehs = list(self.vehs_edges[each].keys())
            for each_veh in pre_observed_vehs:
                if each_veh not in all_vehs:
                    del self.vehs_edges[each][each_veh]
            # update at this step
            for veh in all_vehs:
                self.vehs_edges[each].update({veh: self.get_observed_info_veh(veh, max_speed, max_length, accel_norm)})
            veh_num_per_edge.update({each: len(self.vehs_edges[each].keys()) / w_max})

        # Observed vehicle information
        speeds = []
        accels = []
        dist_to_intersec = []
        edge_number = []
        all_observed_ids = []
        # Traffic light information
        for tl_id, edges in self.mapping_inc.items():  # each intersection
            local_speeds = []
            local_accels = []
            local_dists_to_intersec = []
            local_edge_numbers = []
            for edge in edges:
                # sort to select the closest vehicle
                veh_id_sort = {}
                for veh in self.vehs_edges[edge].keys():
                    veh_id_sort.update({self.vehs_edges[edge][veh][2]: veh})  # closer: larger position
                num_observed = min(self.num_observed, len(self.vehs_edges[edge]))
                observed_ids = [veh_id_sort[sorted(veh_id_sort.keys())[i]] for i in range(0, num_observed)]
                all_observed_ids.extend(observed_ids)

                local_speeds.extend([self.vehs_edges[edge][veh_id][0] for veh_id in observed_ids])
                local_accels.extend([self.vehs_edges[edge][veh_id][1] for veh_id in observed_ids])
                local_dists_to_intersec.extend([self.vehs_edges[edge][veh_id][2] for veh_id in observed_ids])
                local_edge_numbers.extend([self.vehs_edges[edge][veh_id][3] for veh_id in observed_ids])

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    local_speeds.extend([1] * diff)
                    local_accels.extend([0] * diff)
                    local_dists_to_intersec.extend([1] * diff)
                    local_edge_numbers.extend([0] * diff)

            # not 4-leg intersection
            if len(edges) < self.num_local_edges_max:
                diff = self.num_local_edges_max - len(edges)
                local_speeds.extend([1] * diff * self.num_observed)
                local_accels.extend([0] * diff * self.num_observed)
                local_dists_to_intersec.extend([1] * diff * self.num_observed)
                local_edge_numbers.extend([0] * diff * self.num_observed)

            speeds.append(local_speeds)
            accels.append(local_accels)
            dist_to_intersec.append(local_dists_to_intersec)
            edge_number.append(local_edge_numbers)
        self.observed_ids = all_observed_ids

        # the incoming TL for each AV
        incoming_tl = {each: "" for each in self.observed_ids}

        # Traffic light information
        for tl_id in self.k.traffic_light.get_ids():
            tl_id_num = list(self.mapping_inc.keys()).index(tl_id)
            local_edges = self.mapping_inc[tl_id]
            local_edges_out = self.mapping_out[tl_id]

            veh_num_per_in = [veh_num_per_edge[each] for each in local_edges]
            veh_num_per_out = [veh_num_per_edge[each] for each in local_edges_out]
            # not 4-leg intersection
            if len(local_edges) < self.num_local_edges_max:
                diff = self.num_local_edges_max - len(local_edges)
                veh_num_per_in.extend([0] * diff)
            if len(local_edges_out) < self.num_out_edges_max:
                diff = self.num_out_edges_max - len(local_edges_out)
                veh_num_per_out.extend([0] * diff)

            for cav_id in incoming_tl.keys():
                if self.full_name_edge_lane(cav_id) in local_edges:
                    incoming_tl[cav_id] = tl_id_num  # get the id of the approaching TL

            states = self.states_tl[tl_id]
            now_state = self.k.traffic_light.get_state(tl_id)
            state_index = states.index(now_state)

            con = [round(i, 8) for i in np.concatenate(
                [speeds[tl_id_num], accels[tl_id_num], dist_to_intersec[tl_id_num],
                 edge_number[tl_id_num],
                 veh_num_per_in, veh_num_per_out,
                 [self.last_changes[tl_id_num] / 3],
                 [state_index / len(states)], [self.currently_yellows[tl_id_num]]])]

            observation = np.array(con)
            obs.update({tl_id: observation})

            # This is a catch-all for when the relative_node method returns a -1
            # (when there is no node in the direction sought). We add a last
            # item to the lists here, which will serve as a default value.
            self.last_changes.append(0)
            self.currently_yellows.append(1)  # if there is no traffic light

        # agent_CAV information
        for rl_id in self.observed_ids:
            this_pos = self.k.network.edge_length(self.k.vehicle.get_edge(rl_id)) - self.k.vehicle.get_position(
                rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            this_accel = self.k.vehicle.get_realized_accel(rl_id)
            this_accel = (this_accel, -15)[abs(this_accel) >= 15]
            lead_id = self.k.vehicle.get_leader(rl_id)

            this_tl_name = ""
            if incoming_tl[rl_id] != "":
                incoming_tl_id = int(incoming_tl[rl_id])
                this_tl_name = list(self.mapping_inc.keys())[incoming_tl_id]
            else:
                incoming_tl_id = -1  # set default value

            if this_tl_name:
                states = self.states_tl[this_tl_name]
                now_state = self.k.traffic_light.get_state(this_tl_name)
                state_index = states.index(now_state)
            else:
                states = []
                state_index = 0

            if states:
                state_norm = state_index / len(states)
            else:
                state_norm = 0

            if lead_id in ["", None] or self.k.vehicle.get_speed(lead_id) == -1001:
                # in case leader is not visible
                lead_speed = max_speed + this_speed
                lead_head = max_length
                lead_accel = 15
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)
                lead_accel = self.k.vehicle.get_realized_accel(lead_id)
                lead_accel = (lead_accel, -15)[abs(lead_accel) >= 15]

            if lead_head / max_length > 5:
                lead_head = 5 * max_length
            elif lead_head / max_length < -5:
                lead_head = -5 * max_length

            obs.update({rl_id: np.array([
                this_pos / max_length,
                this_speed / max_speed,
                this_accel / max_accel,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                lead_accel / max_accel,
                self.last_changes[incoming_tl_id] / 3,
                state_norm, self.currently_yellows[incoming_tl_id]])})

        self.observation_info = obs
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        for rl_id in self.k.traffic_light.get_ids():
            obs = self.observation_info[rl_id]
            # pressure
            traffic_start = 4 * self.num_local_edges_max * self.num_observed
            inc_traffic = np.sum(obs[traffic_start: traffic_start + self.num_local_edges_max])
            out_traffic = np.sum(obs[traffic_start + self.num_local_edges_max:
                                     traffic_start + self.num_local_edges_max + self.num_out_edges_max])
            reward[rl_id] = -(inc_traffic - out_traffic)

        for rl_id in self.observed_ids:
            edge_id = self.full_name_edge_lane(rl_id)
            veh_ids = list(self.vehs_edges[edge_id].keys())
            reward[rl_id] = rewards.min_delay_edge(self, veh_ids, self.target_speed) - 1 \
                - rewards.stable_acceleration_positive_edge(self, veh_ids)

        return reward

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            if rl_id in self.mapping_inc.keys():
                tl_id_num = list(self.mapping_inc.keys()).index(rl_id)

                # convert values less than 0.0 to zero and above to 1. 0's
                # indicate that we should not switch the direction
                action = rl_action > 0.0

                states = self.states_tl[rl_id]
                now_state = self.k.traffic_light.get_state(rl_id)
                state_index = states.index(now_state)
                if self.currently_yellows[tl_id_num] == 1:  # currently yellow
                    self.last_changes[tl_id_num] += self.sim_step
                    # Check if our timer has exceeded the yellow phase, meaning it should switch
                    if round(float(self.last_changes[tl_id_num]), 8) >= self.min_switch_time:
                        if now_state == states[-1]:
                            state_index = 0
                        else:
                            state_index += 1
                        self.k.traffic_light.set_state(node_id=rl_id, state=states[state_index])
                        if 'G' not in states[state_index]:
                            self.currently_yellows[tl_id_num] = 1
                            self.last_changes[tl_id_num] = 0.0
                        else:
                            self.currently_yellows[tl_id_num] = 0
                else:
                    if action:
                        if now_state == states[-1]:
                            state_index = 0
                        else:
                            state_index += 1
                        self.k.traffic_light.set_state(node_id=rl_id, state=states[state_index])
                        if 'G' not in states[state_index]:
                            self.currently_yellows[tl_id_num] = 1
                            self.last_changes[tl_id_num] = 0.0
                        else:
                            self.currently_yellows[tl_id_num] = 0
            else:
                self.k.vehicle.apply_acceleration(rl_id, rl_actions[rl_id])

    def reset(self, **kwargs):
        self.leader = []
        self.observation_info = {}
        self.vehs_edges = {i: {} for i in self.lanes_related}
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        return super().reset()

    def additional_command(self):
        # specify observed vehicles
        for veh_id in self.k.vehicle.get_ids():
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 255, 255))
        for veh_id in self.observed_ids:
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 0, 0))


class CoTVMixedCustomEnv(CoTVCustomEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_CAVTL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # network, edge
        self.mapping_inc, self.num_local_edges_max, self.mapping_out, self.num_out_edges_max = network.node_mapping
        self.inc_lane = []
        self.lanes_related = []
        for each in self.mapping_inc.values():
            self.lanes_related.extend(each)
            self.inc_lane.extend(each)
        for each in self.mapping_out.values():
            self.lanes_related.extend(each)
        # traffic light
        self.states_tl = network.get_states()
        self.num_traffic_lights = len(self.mapping_inc.keys())
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        # cav
        self.controlled_cav = []
        # traffic - penetration rate
        self.total_veh = env_params.additional_params.get("total_veh", 275)
        self.cav_num = round(env_params.additional_params.get("cav_penetration_rate", 1) * self.total_veh)
        self.veh_type_set = [1] * self.cav_num + [0] * (self.total_veh - self.cav_num)
        self.veh_type = {}
        # vehicle
        self.observation_info = {}
        self.leader = []
        # vehs list for the edges around intersections in the network
        self.vehs_edges = {i: {} for i in self.lanes_related}
        # used during visualization
        self.observed_ids = []
        # exp setting
        self.num_observed = env_params.additional_params.get("num_observed", 1)
        self.min_switch_time = env_params.additional_params["switch_time"]
        self.target_speed = env_params.additional_params.get("target_velocity", 15)

    def choose_cav_agent(self, sorted_veh_id, edge):
        # remove left
        for veh in self.controlled_cav:
            if self.full_name_edge_lane(veh) not in self.inc_lane:
                self.controlled_cav.remove(veh)
        # keep previous
        no_exist = True
        for veh in self.controlled_cav:
            if self.full_name_edge_lane(veh) == edge:
                no_exist = False
                break
        # choose new
        if no_exist:
            if sorted_veh_id:
                for i in range(len(sorted_veh_id)):
                    if sorted_veh_id[i] not in self.veh_type.keys():
                        typee = choice(self.veh_type_set)
                        self.veh_type_set.remove(typee)
                        if typee == 1:
                            self.veh_type.update({sorted_veh_id[i]: typee})
                        else:
                            self.veh_type.update({sorted_veh_id[i]: typee})
                    if self.veh_type[sorted_veh_id[i]] == 1:
                        self.controlled_cav.append(sorted_veh_id[i])
                        break

    def get_state(self):

        obs = {}
        self.leader = []

        # normalizing constants
        max_speed = self.k.network.max_speed()
        edge_length = []
        edge_length.extend([self.k.network.edge_length(edge) for edge in self.k.network.get_edge_list()])
        max_length = max(edge_length)

        max_accel = self.env_params.additional_params["max_accel"]
        max_decel = 15  # emergence stop
        accel_norm = max_accel + max_decel

        veh_lane_pair = {each: [] for each in self.vehs_edges.keys()}
        for each_veh in self.k.vehicle.get_ids():
            # skip the internal links in intersections
            if self.full_name_edge_lane(each_veh) in self.lanes_related:
                veh_lane_pair[self.full_name_edge_lane(each_veh)].append(each_veh)
        # vehicles for incoming and outgoing - info map
        w_max = max_length / 7.5  # normalization for vehicle number, length + min gap
        veh_num_per_edge = {}  # key: name of each edge in the road network
        for each in self.vehs_edges.keys():
            all_vehs = veh_lane_pair[each]
            # remove vehicles already left the edge after one step, not restore at this step
            pre_observed_vehs = list(self.vehs_edges[each].keys())
            for each_veh in pre_observed_vehs:
                if each_veh not in all_vehs:
                    del self.vehs_edges[each][each_veh]
            # update at this step
            for veh in all_vehs:
                self.vehs_edges[each].update({veh: self.get_observed_info_veh(veh, max_speed, max_length, accel_norm)})
            veh_num_per_edge.update({each: len(self.vehs_edges[each].keys()) / w_max})

        # Observed vehicle information
        speeds = []
        accels = []
        dist_to_intersec = []
        edge_number = []
        all_observed_ids = []
        # Traffic light information
        for tl_id, edges in self.mapping_inc.items():  # each intersection
            local_speeds = []
            local_accels = []
            local_dists_to_intersec = []
            local_edge_numbers = []
            for edge in edges:
                # sort to select the closest vehicle
                veh_id_sort = {}
                for veh in self.vehs_edges[edge].keys():
                    veh_id_sort.update({self.vehs_edges[edge][veh][2]: veh})  # closer: larger position
                num_observed = min(self.num_observed, len(self.vehs_edges[edge]))
                sorted_veh_id = []
                for i in range(len(veh_id_sort)):
                    sorted_veh_id.append(veh_id_sort[sorted(veh_id_sort.keys())[i]])
                observed_ids = [sorted_veh_id[i] for i in range(0, num_observed)]
                self.choose_cav_agent(sorted_veh_id, edge)

                all_observed_ids.extend(observed_ids)

                local_speeds.extend([self.vehs_edges[edge][veh_id][0] for veh_id in observed_ids])
                local_accels.extend([self.vehs_edges[edge][veh_id][1] for veh_id in observed_ids])
                local_dists_to_intersec.extend([self.vehs_edges[edge][veh_id][2] for veh_id in observed_ids])
                local_edge_numbers.extend([self.vehs_edges[edge][veh_id][3] for veh_id in observed_ids])

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    local_speeds.extend([1] * diff)
                    local_accels.extend([0] * diff)
                    local_dists_to_intersec.extend([1] * diff)
                    local_edge_numbers.extend([0] * diff)

            # not 4-leg intersection
            if len(edges) < self.num_local_edges_max:
                diff = self.num_local_edges_max - len(edges)
                local_speeds.extend([1] * diff * self.num_observed)
                local_accels.extend([0] * diff * self.num_observed)
                local_dists_to_intersec.extend([1] * diff * self.num_observed)
                local_edge_numbers.extend([0] * diff * self.num_observed)

            speeds.append(local_speeds)
            accels.append(local_accels)
            dist_to_intersec.append(local_dists_to_intersec)
            edge_number.append(local_edge_numbers)
        self.observed_ids = all_observed_ids

        # the incoming TL for each AV
        incoming_tl = {each: "" for each in self.controlled_cav}

        # Traffic light information
        for tl_id in self.k.traffic_light.get_ids():
            tl_id_num = list(self.mapping_inc.keys()).index(tl_id)
            local_edges = self.mapping_inc[tl_id]
            local_edges_out = self.mapping_out[tl_id]

            veh_num_per_in = [veh_num_per_edge[each] for each in local_edges]
            veh_num_per_out = [veh_num_per_edge[each] for each in local_edges_out]
            # not 4-leg intersection
            if len(local_edges) < self.num_local_edges_max:
                diff = self.num_local_edges_max - len(local_edges)
                veh_num_per_in.extend([0] * diff)
            if len(local_edges_out) < self.num_out_edges_max:
                diff = self.num_out_edges_max - len(local_edges_out)
                veh_num_per_out.extend([0] * diff)

            for cav_id in incoming_tl.keys():
                if self.full_name_edge_lane(cav_id) in local_edges:
                    incoming_tl[cav_id] = tl_id_num  # get the id of the approaching TL

            states = self.states_tl[tl_id]
            now_state = self.k.traffic_light.get_state(tl_id)
            state_index = states.index(now_state)

            con = [round(i, 8) for i in np.concatenate(
                [speeds[tl_id_num], accels[tl_id_num], dist_to_intersec[tl_id_num],
                 edge_number[tl_id_num],
                 veh_num_per_in, veh_num_per_out,
                 [self.last_changes[tl_id_num] / 3],
                 [state_index / len(states)], [self.currently_yellows[tl_id_num]]])]

            observation = np.array(con)
            obs.update({tl_id: observation})

            # This is a catch-all for when the relative_node method returns a -1
            # (when there is no node in the direction sought). We add a last
            # item to the lists here, which will serve as a default value.
            self.last_changes.append(0)
            self.currently_yellows.append(1)  # if there is no traffic light

        # agent_CAV information
        for rl_id in self.controlled_cav:
            this_pos = self.k.network.edge_length(self.k.vehicle.get_edge(rl_id)) - self.k.vehicle.get_position(
                rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            this_accel = self.k.vehicle.get_realized_accel(rl_id)
            this_accel = (this_accel, -15)[abs(this_accel) >= 15]
            lead_id = self.k.vehicle.get_leader(rl_id)

            this_tl_name = ""
            if incoming_tl[rl_id] != "":
                incoming_tl_id = int(incoming_tl[rl_id])
                this_tl_name = list(self.mapping_inc.keys())[incoming_tl_id]
            else:
                incoming_tl_id = -1  # set default value

            if this_tl_name:
                states = self.states_tl[this_tl_name]
                now_state = self.k.traffic_light.get_state(this_tl_name)
                state_index = states.index(now_state)
            else:
                states = []
                state_index = 0

            if states:
                state_norm = state_index / len(states)
            else:
                state_norm = 0

            if lead_id in ["", None] or self.k.vehicle.get_speed(lead_id) == -1001:
                # in case leader is not visible
                lead_speed = max_speed + this_speed
                lead_head = max_length
                lead_accel = 15
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)
                lead_accel = self.k.vehicle.get_realized_accel(lead_id)
                lead_accel = (lead_accel, -15)[abs(lead_accel) >= 15]

            if lead_head / max_length > 5:
                lead_head = 5 * max_length
            elif lead_head / max_length < -5:
                lead_head = -5 * max_length

            obs.update({rl_id: np.array([
                this_pos / max_length,
                this_speed / max_speed,
                this_accel / max_accel,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                lead_accel / max_accel,
                self.last_changes[incoming_tl_id] / 3,
                state_norm, self.currently_yellows[incoming_tl_id]])})

        self.observation_info = obs
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        for rl_id in self.k.traffic_light.get_ids():
            obs = self.observation_info[rl_id]
            # pressure
            traffic_start = 4 * self.num_local_edges_max * self.num_observed
            inc_traffic = np.sum(obs[traffic_start: traffic_start + self.num_local_edges_max])
            out_traffic = np.sum(obs[traffic_start + self.num_local_edges_max:
                                     traffic_start + self.num_local_edges_max + self.num_out_edges_max])
            reward[rl_id] = -(inc_traffic - out_traffic)

        for rl_id in self.controlled_cav:
            edge_id = self.full_name_edge_lane(rl_id)
            veh_ids = list(self.vehs_edges[edge_id].keys())
            reward[rl_id] = rewards.min_delay_edge(self, veh_ids, self.target_speed) - 1 \
                            - rewards.stable_acceleration_positive_edge(self, veh_ids)

        return reward

    def reset(self, **kwargs):
        self.leader = []
        self.observation_info = {}
        self.vehs_edges = {i: {} for i in self.lanes_related}
        self.controlled_cav = []
        self.veh_type_set = [1] * self.cav_num + [0] * (self.total_veh - self.cav_num)
        self.veh_type = {}
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        return super().reset()

    def additional_command(self):
        # specify observed vehicles
        for veh_id in self.k.vehicle.get_ids():
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 255, 255))
        for veh_id in self.controlled_cav:
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 0, 0))


class CoTVAllCustomEnv(CoTVCustomEnv):
    """
        To adopt network from sumo
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_CAVTL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # network, edge
        self.mapping_inc, self.num_local_edges_max, self.mapping_out, self.num_out_edges_max = network.node_mapping
        self.all_lane = network.specify_edges()
        # traffic light
        self.states_tl = network.get_states()
        self.num_traffic_lights = len(self.mapping_inc.keys())
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        # vehicle
        self.observation_info = {}
        self.leader = []
        # vehs list for all the edges in the network
        self.vehs_edges = {i: {} for i in self.all_lane}
        # used during visualization
        self.observed_ids = []
        # exp setting
        self.num_observed = env_params.additional_params.get("num_observed", 1)
        self.min_switch_time = env_params.additional_params["switch_time"]
        self.target_speed = env_params.additional_params.get("target_velocity", 15)

    def convert_edge_into_num(self, edge_id):
        return self.all_lane.index(edge_id) + 1

    def get_observed_info_veh(self, veh_id, max_speed, max_length, accel_norm):
        speed_veh = self.k.vehicle.get_speed(veh_id) / max_speed
        accel_veh = self.k.vehicle.get_realized_accel(veh_id)
        dis_veh = (self.k.network.edge_length(self.k.vehicle.get_edge(veh_id)) -
                   self.k.vehicle.get_position(veh_id)) / max_length
        edge_veh = self.convert_edge_into_num(self.full_name_edge_lane(veh_id)) / len(self.all_lane)
        # 0: no road to form a 4-leg intersection

        # accel normalization
        if accel_veh < -15:
            accel_veh = -15
        accel_veh = (accel_veh + 15) / accel_norm
        return [speed_veh, accel_veh, dis_veh, edge_veh]

    def get_state(self):

        obs = {}
        self.leader = []

        # normalizing constants
        max_speed = self.k.network.max_speed()
        edge_length = []
        edge_length.extend([self.k.network.edge_length(edge) for edge in self.k.network.get_edge_list()])
        max_length = max(edge_length)

        max_accel = self.env_params.additional_params["max_accel"]
        max_decel = 15  # emergence stop
        accel_norm = max_accel + max_decel

        self.vehs_edges = {i: {} for i in self.all_lane}  # differ from CoTVCustomEnv
        veh_lane_pair = {each: [] for each in self.vehs_edges.keys()}
        for each_veh in self.k.vehicle.get_ids():
            # skip the internal links in intersections
            if self.full_name_edge_lane(each_veh) in self.all_lane:
                veh_lane_pair[self.full_name_edge_lane(each_veh)].append(each_veh)
        # vehicles for incoming and outgoing - info map
        w_max = max_length / 7.5  # normalization for vehicle number, length + min gap
        veh_num_per_edge = {}  # key: name of each edge in the road network
        for each in self.vehs_edges.keys():
            all_vehs = veh_lane_pair[each]
            # remove vehicles already left the edge after one step, not restore at this step
            pre_observed_vehs = list(self.vehs_edges[each].keys())
            for each_veh in pre_observed_vehs:
                if each_veh not in all_vehs:
                    del self.vehs_edges[each][each_veh]
            # update at this step
            for veh in all_vehs:
                self.vehs_edges[each].update({veh: self.get_observed_info_veh(veh, max_speed, max_length, accel_norm)})
            veh_num_per_edge.update({each: len(self.vehs_edges[each].keys()) / w_max})

        # Observed vehicle information
        speeds = []
        accels = []
        dist_to_intersec = []
        edge_number = []
        all_observed_ids = []
        # Traffic light information
        for tl_id, edges in self.mapping_inc.items():  # each intersection
            local_speeds = []
            local_accels = []
            local_dists_to_intersec = []
            local_edge_numbers = []
            for edge in edges:
                # sort to select the closest vehicle
                veh_id_sort = {}
                for veh in self.vehs_edges[edge].keys():
                    veh_id_sort.update({self.vehs_edges[edge][veh][2]: veh})  # closer: larger position
                num_observed = min(self.num_observed, len(self.vehs_edges[edge]))
                observed_ids = [veh_id_sort[sorted(veh_id_sort.keys())[i]] for i in range(0, num_observed)]
                all_observed_ids.extend(observed_ids)

                local_speeds.extend([self.vehs_edges[edge][veh_id][0] for veh_id in observed_ids])
                local_accels.extend([self.vehs_edges[edge][veh_id][1] for veh_id in observed_ids])
                local_dists_to_intersec.extend([self.vehs_edges[edge][veh_id][2] for veh_id in observed_ids])
                local_edge_numbers.extend([self.vehs_edges[edge][veh_id][3] for veh_id in observed_ids])

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    local_speeds.extend([1] * diff)
                    local_accels.extend([0] * diff)
                    local_dists_to_intersec.extend([1] * diff)
                    local_edge_numbers.extend([0] * diff)

            # not 4-leg intersection
            if len(edges) < self.num_local_edges_max:
                diff = self.num_local_edges_max - len(edges)
                local_speeds.extend([1] * diff * self.num_observed)
                local_accels.extend([0] * diff * self.num_observed)
                local_dists_to_intersec.extend([1] * diff * self.num_observed)
                local_edge_numbers.extend([0] * diff * self.num_observed)

            speeds.append(local_speeds)
            accels.append(local_accels)
            dist_to_intersec.append(local_dists_to_intersec)
            edge_number.append(local_edge_numbers)
        self.observed_ids = all_observed_ids

        # the incoming TL for each AV
        incoming_tl = {each: "" for each in self.k.vehicle.get_ids()}

        # Traffic light information
        for tl_id in self.k.traffic_light.get_ids():
            tl_id_num = list(self.mapping_inc.keys()).index(tl_id)
            local_edges = self.mapping_inc[tl_id]
            local_edges_out = self.mapping_out[tl_id]

            veh_num_per_in = [veh_num_per_edge[each] for each in local_edges]
            veh_num_per_out = [veh_num_per_edge[each] for each in local_edges_out]
            # not 4-leg intersection
            if len(local_edges) < self.num_local_edges_max:
                diff = self.num_local_edges_max - len(local_edges)
                veh_num_per_in.extend([0] * diff)
            if len(local_edges_out) < self.num_out_edges_max:
                diff = self.num_out_edges_max - len(local_edges_out)
                veh_num_per_out.extend([0] * diff)

            for cav_id in incoming_tl.keys():
                if self.full_name_edge_lane(cav_id) in local_edges:
                    incoming_tl[cav_id] = tl_id_num  # get the id of the approaching TL

            states = self.states_tl[tl_id]
            now_state = self.k.traffic_light.get_state(tl_id)
            state_index = states.index(now_state)

            con = [round(i, 8) for i in np.concatenate(
                [speeds[tl_id_num], accels[tl_id_num], dist_to_intersec[tl_id_num],
                 edge_number[tl_id_num],
                 veh_num_per_in, veh_num_per_out,
                 [self.last_changes[tl_id_num] / 3],
                 [state_index / len(states)], [self.currently_yellows[tl_id_num]]])]

            observation = np.array(con)
            obs.update({tl_id: observation})

            # This is a catch-all for when the relative_node method returns a -1
            # (when there is no node in the direction sought). We add a last
            # item to the lists here, which will serve as a default value.
            self.last_changes.append(0)
            self.currently_yellows.append(1)  # if there is no traffic light

        # agent_CAV information
        for rl_id in self.k.vehicle.get_ids():
            this_pos = self.k.network.edge_length(self.k.vehicle.get_edge(rl_id)) - self.k.vehicle.get_position(
                rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            this_accel = self.k.vehicle.get_realized_accel(rl_id)
            this_accel = (this_accel, -15)[abs(this_accel) >= 15]
            lead_id = self.k.vehicle.get_leader(rl_id)

            this_tl_name = ""
            if incoming_tl[rl_id] != "":
                incoming_tl_id = int(incoming_tl[rl_id])
                this_tl_name = list(self.mapping_inc.keys())[incoming_tl_id]
            else:
                incoming_tl_id = -1  # set default value

            if this_tl_name:
                states = self.states_tl[this_tl_name]
                now_state = self.k.traffic_light.get_state(this_tl_name)
                state_index = states.index(now_state)
            else:
                states = []
                state_index = 0

            if states:
                state_norm = state_index / len(states)
            else:
                state_norm = 0

            if lead_id in ["", None] or self.k.vehicle.get_speed(lead_id) == -1001:
                # in case leader is not visible
                lead_speed = max_speed + this_speed
                lead_head = max_length
                lead_accel = 15
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)
                lead_accel = self.k.vehicle.get_realized_accel(lead_id)
                lead_accel = (lead_accel, -15)[abs(lead_accel) >= 15]

            if lead_head / max_length > 5:
                lead_head = 5 * max_length
            elif lead_head / max_length < -5:
                lead_head = -5 * max_length

            obs.update({rl_id: np.array([
                this_pos / max_length,
                this_speed / max_speed,
                this_accel / max_accel,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                lead_accel / max_accel,
                self.last_changes[incoming_tl_id] / 3,
                state_norm, self.currently_yellows[incoming_tl_id]])})

        self.observation_info = obs
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        for rl_id in self.k.traffic_light.get_ids():
            obs = self.observation_info[rl_id]
            # pressure
            traffic_start = 4 * self.num_local_edges_max * self.num_observed
            inc_traffic = np.sum(obs[traffic_start: traffic_start + self.num_local_edges_max])
            out_traffic = np.sum(obs[traffic_start + self.num_local_edges_max:
                                     traffic_start + self.num_local_edges_max + self.num_out_edges_max])
            reward[rl_id] = -(inc_traffic - out_traffic)

        for rl_id in self.k.vehicle.get_ids():
            edge_id = self.full_name_edge_lane(rl_id)
            veh_ids = self.k.vehicle.get_ids_by_edge(edge_id)  # list(self.vehs_edges[edge_id].keys())
            reward[rl_id] = rewards.min_delay_edge(self, veh_ids, self.target_speed) - 1 \
                            - rewards.stable_acceleration_positive_edge(self, veh_ids)
        return reward

    def additional_command(self):
        # specify observed vehicles
        for veh_id in self.k.vehicle.get_ids():
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 0, 0))


class PressLightCustomEnv(MultiEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_TL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # network, edge
        self.mapping_inc, self.num_local_edges_max, self.mapping_out, self.num_out_edges_max = network.node_mapping
        self.lanes_related = []
        self.lanes_split = []
        for each in self.mapping_inc.values():
            self.lanes_related.extend(each)
            self.lanes_split.extend(each)
        for each in self.mapping_out.values():
            self.lanes_related.extend(each)
        self.lanes_related = list(set(self.lanes_related))
        # normalizing constants
        edge_length = []
        edge_length.extend([self.k.network.edge_length(edge) for edge in self.k.network.get_edge_list()])
        self.max_length = max(edge_length)
        # traffic light
        self.num_traffic_lights = len(self.mapping_inc.keys())
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        self.state_tl = network.get_states()
        self.min_switch_time = env_params.additional_params["switch_time"]
        self.static = self.env_params.additional_params.get("static", False)
        # obs
        self.observation_info = {}

    @property
    def action_space(self):
        return Discrete(2)

    @property
    def observation_space(self):
        return Box(low=0., high=math.ceil(self.max_length / 7.5),
                   shape=(self.num_local_edges_max*3 + self.num_out_edges_max + 1,))

    def full_name_edge_lane(self, veh_id):
        edge_id = self.k.vehicle.get_edge(veh_id)
        lane_id = self.k.vehicle.get_lane(veh_id)
        return edge_id + '_' + str(lane_id)

    def get_state(self, **kwargs):
        obs = {}

        veh_lane_pair = {each: [] for each in self.lanes_related}
        for each_veh in self.k.vehicle.get_ids():
            # skip the internal links in intersections
            if self.full_name_edge_lane(each_veh) in self.lanes_related:
                veh_lane_pair[self.full_name_edge_lane(each_veh)].append(each_veh)

        veh_num_per_edge = {}  # {name of each edge: road occupancy}
        veh_num_per_edge_split = {}  # {name of each edge segment: #veh}
        veh_num_per_edge_real = {}  # {name of each edge: #veh}
        for lane, veh_list in veh_lane_pair.items():
            w_nor = math.ceil(self.k.network.edge_length(lane.split('_')[0]) / 7.5)
            veh_num_per_edge.update({lane: len(veh_list) / w_nor})
            veh_num_per_edge_real.update({lane: len(veh_list)})
            if lane in self.lanes_split:
                for i in range(0, 3):
                    veh_num_per_edge_split.update({lane + ':' + str(i): 0})
                for each_veh in veh_list:
                    dis_veh = (self.k.network.edge_length(lane.split('_')[0]) -
                               self.k.vehicle.get_position(each_veh)) / \
                              self.k.network.edge_length(lane.split('_')[0])
                    if dis_veh < 0.34:
                        veh_num_per_edge_split[lane + ':0'] += 1
                    elif dis_veh > 0.66:
                        veh_num_per_edge_split[lane + ':2'] += 1
                    else:
                        veh_num_per_edge_split[lane + ':1'] += 1

        # Traffic light information
        for tl_id in self.k.traffic_light.get_ids():
            local_edges = self.mapping_inc[tl_id]
            local_edges_out = self.mapping_out[tl_id]

            veh_num_per_in_0 = [veh_num_per_edge_split[each + ':0'] for each in local_edges]
            veh_num_per_in_1 = [veh_num_per_edge_split[each + ':1'] for each in local_edges]
            veh_num_per_in_2 = [veh_num_per_edge_split[each + ':2'] for each in local_edges]
            veh_num_per_out = [veh_num_per_edge_real[each] for each in local_edges_out]

            veh_num_per_in_nor = [veh_num_per_edge[each] for each in local_edges]
            veh_num_per_out_nor = [veh_num_per_edge[each] for each in local_edges_out]
            # not 4-leg intersection
            if len(local_edges) < self.num_local_edges_max:
                diff = self.num_local_edges_max - len(local_edges)
                veh_num_per_in_0.extend([0] * diff)
                veh_num_per_in_1.extend([0] * diff)
                veh_num_per_in_2.extend([0] * diff)
                veh_num_per_in_nor.extend([0] * diff)
            if len(local_edges_out) < self.num_out_edges_max:
                diff = self.num_out_edges_max - len(local_edges_out)
                veh_num_per_out.extend([0] * diff)
                veh_num_per_out_nor.extend([0]*diff)

            states = self.state_tl[tl_id]
            now_state = self.k.traffic_light.get_state(tl_id)
            state_index = states.index(now_state)

            con = [round(i, 8) for i in np.concatenate(
                [veh_num_per_in_0, veh_num_per_in_1, veh_num_per_in_2, veh_num_per_out, [state_index]])]

            observation = np.array(con)
            obs.update({tl_id: observation})
            self.observation_info.update({tl_id: np.array([round(i, 8) for i in np.concatenate(
                [veh_num_per_in_nor, veh_num_per_out_nor])])})

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        for rl_id in self.k.traffic_light.get_ids():
            obs = self.observation_info[rl_id]
            w_l_m = []
            for i in range(0, self.num_local_edges_max):
                w_l_m.append(obs[i] - obs[i + self.num_local_edges_max])

            reward[rl_id] = -np.abs(np.sum(np.array(w_l_m)))

        return reward

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            tl_id_num = list(self.mapping_inc.keys()).index(rl_id)

            # convert values less than 0.0 to zero and above to 1. 0's
            # indicate that we should not switch the direction
            action = rl_action > 0.0

            non_rl = []
            if self.static:
                non_rl = [i for i in range(0, len(rl_actions.keys()))]

            states = self.state_tl[rl_id]
            now_state = self.k.traffic_light.get_state(rl_id)
            state_index = states.index(now_state)
            if self.currently_yellows[tl_id_num] == 1:  # currently yellow
                self.last_changes[tl_id_num] += self.sim_step
                # Check if our timer has exceeded the yellow phase, meaning it should switch
                if round(float(self.last_changes[tl_id_num]), 8) >= self.min_switch_time:
                    if tl_id_num not in non_rl:
                        if now_state == states[-1]:
                            state_index = 0
                        else:
                            state_index += 1
                        self.k.traffic_light.set_state(node_id=rl_id, state=states[state_index])
                        if 'G' not in states[state_index]:
                            self.currently_yellows[tl_id_num] = 1
                            self.last_changes[tl_id_num] = 0.0
                        else:
                            self.currently_yellows[tl_id_num] = 0
            else:
                if action:
                    if tl_id_num not in non_rl:
                        if now_state == states[-1]:
                            state_index = 0
                        else:
                            state_index += 1
                        self.k.traffic_light.set_state(node_id=rl_id, state=states[state_index])
                        if 'G' not in states[state_index]:
                            self.currently_yellows[tl_id_num] = 1
                            self.last_changes[tl_id_num] = 0.0
                        else:
                            self.currently_yellows[tl_id_num] = 0

    def reset(self, **kwargs):
        self.observation_info = {}
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        return super().reset()


class FlowCAVCustomEnv(MultiEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        for p in ADDITIONAL_ENV_PARAMS_CAV.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # network, edge
        self.mapping_inc, self.num_local_edges_max, self.mapping_out, self.num_out_edges_max = network.node_mapping
        self.lanes_related = []
        for each in self.mapping_inc.values():
            self.lanes_related.extend(each)
        # traffic light
        self.states_tl = network.get_states()
        self.num_traffic_lights = len(self.mapping_inc.keys())
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        # vehicle
        self.leader = []
        self.controlled_cav = []
        # exp setting
        self.num_observed = env_params.additional_params.get("num_observed", 1)
        self.target_speed = env_params.additional_params.get("target_velocity", 15)

    @property
    def action_space(self):
        return Box(low=-abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"], shape=(1,))

    @property
    def observation_space(self):
        return Box(low=-5, high=5, shape=(3,))

    def full_name_edge_lane(self, veh_id):
        edge_id = self.k.vehicle.get_edge(veh_id)
        lane_id = self.k.vehicle.get_lane(veh_id)
        return edge_id + '_' + str(lane_id)

    def specify_cav(self, max_length):
        controlled_cavs = []
        veh_lane_pair = {each: [] for each in self.lanes_related}
        for each_veh in self.k.vehicle.get_ids():
            # skip the internal links in intersections
            if self.full_name_edge_lane(each_veh) in self.lanes_related:
                veh_lane_pair[self.full_name_edge_lane(each_veh)].append(each_veh)

        for edge in self.lanes_related:
            veh_id_sort = {}
            for veh in veh_lane_pair[edge]:
                veh_id_sort.update({(self.k.network.edge_length(self.k.vehicle.get_edge(veh)) -
                                     self.k.vehicle.get_position(veh)) / max_length: veh})  # closer: larger position
            num_observed = min(self.num_observed, len(veh_lane_pair[edge]))
            closest_veh = [veh_id_sort[sorted(veh_id_sort.keys())[i]] for i in range(0, num_observed)]
            controlled_cavs.extend(closest_veh)
        return controlled_cavs

    def get_state(self, **kwargs):
        self.leader = []
        obs = {}

        # normalizing constants
        max_speed = self.k.network.max_speed()
        edge_length = []
        edge_length.extend([self.k.network.edge_length(edge) for edge in self.k.network.get_edge_list()])
        max_length = max(edge_length)

        self.controlled_cav = self.specify_cav(max_length)

        for rl_id in self.controlled_cav:
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)

            if lead_id in ["", None] or self.k.vehicle.get_speed(lead_id) == -1001:
                # in case leader is not visible -> vehicles cross intersection in conflicting direction; red light
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)
                # negative if lead is vehicles cross intersection in conflicting direction

            if lead_head / max_length > 5:
                lead_head = 5 * max_length
            elif lead_head / max_length < -5:
                lead_head = -5 * max_length

            # Add the next observation.
            obs[rl_id] = np.array([
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
            ])
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        # for global reward per vehicle
        if rl_actions is None:
            return {}

        reward = {}

        all_speed = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(all_speed < -100):
            return {}

        # reward average velocity
        eta_2 = 4.  # 4.
        reward_value = eta_2 * np.mean(all_speed) / 20

        # punish accelerations (should lead to reduced stop-and-go waves)
        eta = 4  # 0.25
        all_accel = np.array([
            self.k.vehicle.get_realized_accel(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        mean_accel = np.mean(np.abs(all_accel))
        accel_threshold = 0

        if mean_accel > accel_threshold:
            reward_value += eta * (accel_threshold - mean_accel)

        for rl_id in self.controlled_cav:  # self.k.vehicle.get_rl_ids()
            reward.update({rl_id: reward_value})
        return reward

    def _apply_rl_actions(self, rl_actions):
        for veh_id in self.controlled_cav:
            self.k.vehicle.apply_acceleration(veh_id, rl_actions[veh_id])

    def reset(self, **kwargs):
        self.leader = []
        self.controlled_cav = []
        return super().reset()

    def additional_command(self):
        # specify observed vehicles
        for veh_id in self.k.vehicle.get_ids():
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 255, 255))
        for veh_id in self.controlled_cav:
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 0, 0))


class CoTVNOCoorCustomEnv(MultiEnv):
    """
        To adopt network from sumo
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_CAVTL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # network, edge
        self.mapping_inc, self.num_local_edges_max, self.mapping_out, self.num_out_edges_max = network.node_mapping
        self.lanes_related = []
        for each in self.mapping_inc.values():
            self.lanes_related.extend(each)
        for each in self.mapping_out.values():
            self.lanes_related.extend(each)
        # traffic light
        self.states_tl = network.get_states()
        self.num_traffic_lights = len(self.mapping_inc.keys())
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        # vehicle
        self.observation_info = {}
        self.leader = []
        # vehs list for the edges around intersections in the network
        self.vehs_edges = {i: {} for i in self.lanes_related}
        # used during visualization
        self.observed_ids = []
        # exp setting
        self.num_observed = env_params.additional_params.get("num_observed", 1)
        self.min_switch_time = env_params.additional_params["switch_time"]
        self.target_speed = env_params.additional_params.get("target_velocity", 15)

    @property
    def action_space_tl(self):
        return Discrete(2)

    @property
    def action_space_av(self):
        return Box(low=-abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"], shape=(1,))

    @property
    def observation_space_tl(self):
        return Box(low=0., high=1, shape=(self.num_local_edges_max + self.num_out_edges_max + 3,))

    @property
    def observation_space_av(self):
        return Box(low=-5, high=5, shape=(6,))

    def full_name_edge_lane(self, veh_id):
        edge_id = self.k.vehicle.get_edge(veh_id)
        lane_id = self.k.vehicle.get_lane(veh_id)
        return edge_id + '_' + str(lane_id)

    def convert_edge_into_num(self, edge_id):
        return self.lanes_related.index(edge_id) + 1

    def get_observed_info_veh(self, veh_id, max_length):
        return (self.k.network.edge_length(self.k.vehicle.get_edge(veh_id)) -
                self.k.vehicle.get_position(veh_id)) / max_length

    def get_state(self):

        obs = {}
        self.leader = []

        # normalizing constants
        max_speed = self.k.network.max_speed()
        edge_length = []
        edge_length.extend([self.k.network.edge_length(edge) for edge in self.k.network.get_edge_list()])
        max_length = max(edge_length)

        max_accel = self.env_params.additional_params["max_accel"]
        max_decel = 15  # emergence stop
        accel_norm = max_accel + max_decel

        veh_lane_pair = {each: [] for each in self.vehs_edges.keys()}
        for each_veh in self.k.vehicle.get_ids():
            # skip the internal links in intersections
            if self.full_name_edge_lane(each_veh) in self.lanes_related:
                veh_lane_pair[self.full_name_edge_lane(each_veh)].append(each_veh)
        # vehicles for incoming and outgoing - info map
        w_max = max_length / 7.5  # normalization for vehicle number, length + min gap
        veh_num_per_edge = {}  # key: name of each edge in the road network
        for each in self.vehs_edges.keys():
            all_vehs = veh_lane_pair[each]
            # remove vehicles already left the edge after one step, not restore at this step
            pre_observed_vehs = list(self.vehs_edges[each].keys())
            for each_veh in pre_observed_vehs:
                if each_veh not in all_vehs:
                    del self.vehs_edges[each][each_veh]
            # update at this step
            for veh in all_vehs:
                self.vehs_edges[each].update({veh: self.get_observed_info_veh(veh, max_length)})
            veh_num_per_edge.update({each: len(self.vehs_edges[each].keys()) / w_max})

        # Observed vehicle information
        all_observed_ids = []
        # Traffic light information
        for tl_id, edges in self.mapping_inc.items():  # each intersection
            for edge in edges:
                # sort to select the closest vehicle
                veh_id_sort = {}
                for veh in self.vehs_edges[edge].keys():
                    veh_id_sort.update({self.vehs_edges[edge][veh]: veh})  # closer: larger position
                num_observed = min(self.num_observed, len(self.vehs_edges[edge]))
                observed_ids = [veh_id_sort[sorted(veh_id_sort.keys())[i]] for i in range(0, num_observed)]
                all_observed_ids.extend(observed_ids)

        self.observed_ids = all_observed_ids

        # Traffic light information
        for tl_id in self.k.traffic_light.get_ids():
            tl_id_num = list(self.mapping_inc.keys()).index(tl_id)
            local_edges = self.mapping_inc[tl_id]
            local_edges_out = self.mapping_out[tl_id]

            veh_num_per_in = [veh_num_per_edge[each] for each in local_edges]
            veh_num_per_out = [veh_num_per_edge[each] for each in local_edges_out]
            # not 4-leg intersection
            if len(local_edges) < self.num_local_edges_max:
                diff = self.num_local_edges_max - len(local_edges)
                veh_num_per_in.extend([0] * diff)
            if len(local_edges_out) < self.num_out_edges_max:
                diff = self.num_out_edges_max - len(local_edges_out)
                veh_num_per_out.extend([0] * diff)

            states = self.states_tl[tl_id]
            now_state = self.k.traffic_light.get_state(tl_id)
            state_index = states.index(now_state)

            con = [round(i, 8) for i in np.concatenate(
                [veh_num_per_in, veh_num_per_out,
                 [self.last_changes[tl_id_num] / 3],
                 [state_index / len(states)], [self.currently_yellows[tl_id_num]]])]

            observation = np.array(con)
            obs.update({tl_id: observation})

            # This is a catch-all for when the relative_node method returns a -1
            # (when there is no node in the direction sought). We add a last
            # item to the lists here, which will serve as a default value.
            self.last_changes.append(0)
            self.currently_yellows.append(1)  # if there is no traffic light

        # agent_CAV information
        for rl_id in self.observed_ids:
            this_pos = self.k.network.edge_length(self.k.vehicle.get_edge(rl_id)) - self.k.vehicle.get_position(
                rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            this_accel = self.k.vehicle.get_realized_accel(rl_id)
            this_accel = (this_accel, -15)[abs(this_accel) >= 15]
            lead_id = self.k.vehicle.get_leader(rl_id)

            if lead_id in ["", None] or self.k.vehicle.get_speed(lead_id) == -1001:
                # in case leader is not visible
                lead_speed = max_speed + this_speed
                lead_head = max_length
                lead_accel = 15
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)
                lead_accel = self.k.vehicle.get_realized_accel(lead_id)
                lead_accel = (lead_accel, -15)[abs(lead_accel) >= 15]

            if lead_head / max_length > 5:
                lead_head = 5 * max_length
            elif lead_head / max_length < -5:
                lead_head = -5 * max_length

            obs.update({rl_id: np.array([
                this_pos / max_length,
                this_speed / max_speed,
                this_accel / max_accel,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                lead_accel / max_accel])})

        self.observation_info = obs
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        for rl_id in self.k.traffic_light.get_ids():
            obs = self.observation_info[rl_id]
            # pressure
            inc_traffic = np.sum(obs[:self.num_local_edges_max])
            out_traffic = np.sum(obs[self.num_local_edges_max:self.num_local_edges_max + self.num_out_edges_max])
            reward[rl_id] = -(inc_traffic - out_traffic)

        for rl_id in self.observed_ids:
            edge_id = self.full_name_edge_lane(rl_id)
            veh_ids = list(self.vehs_edges[edge_id].keys())
            reward[rl_id] = rewards.min_delay_edge(self, veh_ids, self.target_speed) - 1 \
                - rewards.stable_acceleration_positive_edge(self, veh_ids)

        return reward

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            if rl_id in self.mapping_inc.keys():
                tl_id_num = list(self.mapping_inc.keys()).index(rl_id)

                # convert values less than 0.0 to zero and above to 1. 0's
                # indicate that we should not switch the direction
                action = rl_action > 0.0

                states = self.states_tl[rl_id]
                now_state = self.k.traffic_light.get_state(rl_id)
                state_index = states.index(now_state)
                if self.currently_yellows[tl_id_num] == 1:  # currently yellow
                    self.last_changes[tl_id_num] += self.sim_step
                    # Check if our timer has exceeded the yellow phase, meaning it should switch
                    if round(float(self.last_changes[tl_id_num]), 8) >= self.min_switch_time:
                        if now_state == states[-1]:
                            state_index = 0
                        else:
                            state_index += 1
                        self.k.traffic_light.set_state(node_id=rl_id, state=states[state_index])
                        if 'G' not in states[state_index]:
                            self.currently_yellows[tl_id_num] = 1
                            self.last_changes[tl_id_num] = 0.0
                        else:
                            self.currently_yellows[tl_id_num] = 0
                else:
                    if action:
                        if now_state == states[-1]:
                            state_index = 0
                        else:
                            state_index += 1
                        self.k.traffic_light.set_state(node_id=rl_id, state=states[state_index])
                        if 'G' not in states[state_index]:
                            self.currently_yellows[tl_id_num] = 1
                            self.last_changes[tl_id_num] = 0.0
                        else:
                            self.currently_yellows[tl_id_num] = 0
            else:
                self.k.vehicle.apply_acceleration(rl_id, rl_actions[rl_id])

    def reset(self, **kwargs):
        self.leader = []
        self.observation_info = {}
        self.vehs_edges = {i: {} for i in self.lanes_related}
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        return super().reset()

    def additional_command(self):
        # specify observed vehicles
        for veh_id in self.k.vehicle.get_ids():
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 255, 255))
        for veh_id in self.observed_ids:
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 0, 0))


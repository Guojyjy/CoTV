from flow.networks import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
import sumolib

ADDITIONAL_NET_PARAMS = {
    "controlled_intersections": []
}


class SumoNetwork(Network):

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):

        self.net_file_path = net_params.template["net"]

        super().__init__(name, vehicles, net_params, initial_config, traffic_lights)

    def specify_edges(self):
        # all except for internal links, more than roads included in intersection mapping
        edges = []
        for edge in sumolib.output.parse(self.net_file_path, ['edge']):
            if not edge.function:
                for lane in edge['lane']:
                    edges.append(lane.id)
        return edges

    @property
    def node_mapping(self):
        """
        Map TL with edges

        Returns a list of each incoming edge for the corresponding TL, and the maximum edge number
        a list of each outgoing for the corresponding TL, and the maximum edge number
        -------

        """
        mapping_inc = {}
        num_local_edges_max = 0
        for junction in sumolib.output.parse(self.net_file_path, ['junction']):
            if junction.type == 'traffic_light':
                inc_edges = junction.incLanes.split(' ')
                mapping_inc.update({junction.id: inc_edges})
                num_local_edges_max = (len(inc_edges), num_local_edges_max)[num_local_edges_max > len(inc_edges)]

        mapping_out = {each: [] for each in mapping_inc.keys()}
        num_out_edges_max = 0
        for tl_id, inc_edges in mapping_inc.items():
            out_edges = []
            for connection in sumolib.output.parse(self.net_file_path, ['connection']):
                if connection.attr_from+'_'+connection.fromLane in inc_edges:
                    out_edges.append(connection.to+'_'+connection.toLane)
            mapping_out[tl_id] = list(set(out_edges))
            num_out_edges_max = (len(mapping_out[tl_id]), num_out_edges_max)[num_out_edges_max
                                                                             > len(mapping_out[tl_id])]
        return mapping_inc, num_local_edges_max, mapping_out, num_out_edges_max

    def get_states(self):
        states_tl = {}
        for tlLogic in sumolib.output.parse(self.net_file_path, ['tlLogic']):
            states = []
            for each in tlLogic['phase']:
                states.append(each.state)
            states_tl.update({tlLogic.id: states})
        return states_tl


class UAVNetwork(Network):

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):

        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.net_file_path = net_params.template["net"]
        self.controlled_tl = net_params.additional_params.get("controlled_intersections")

        super().__init__(name, vehicles, net_params, initial_config, traffic_lights)

    def node_mapping_choose(self, controlled_tl):
        """
        Map the specific TLs with edges

        Returns a list of each incoming edge for the corresponding TL, and the maximum edge number
        a list of each outgoing for the corresponding TL, and the maximum edge number
        -------

        """
        mapping_inc = {}
        num_local_edges_max = 0
        for junction in sumolib.output.parse(self.net_file_path, ['junction']):
            if junction.type == 'traffic_light' and junction.id in controlled_tl:
                inc_edges = junction.incLanes.split(' ')
                mapping_inc.update({junction.id: inc_edges})
                num_local_edges_max = (len(inc_edges), num_local_edges_max)[num_local_edges_max > len(inc_edges)]

        mapping_out = {each: [] for each in mapping_inc.keys()}
        num_out_edges_max = 0
        for tl_id, inc_edges in mapping_inc.items():
            out_edges = []
            for connection in sumolib.output.parse(self.net_file_path, ['connection']):
                if connection.attr_from + '_' + connection.fromLane in inc_edges:
                    out_edges.append(connection.to + '_' + connection.toLane)
            mapping_out[tl_id] = list(set(out_edges))
            num_out_edges_max = (len(mapping_out[tl_id]), num_out_edges_max)[num_out_edges_max
                                                                             > len(mapping_out[tl_id])]
        # return sorted(mapping.items(), key=lambda x: x[0])
        return mapping_inc, num_local_edges_max, mapping_out, num_out_edges_max

    def get_states_choose(self, controlled_tl):
        states_tl = {}
        for tlLogic in sumolib.output.parse(self.net_file_path, ['tlLogic']):
            if tlLogic.id in controlled_tl:
                states = []
                for each in tlLogic['phase']:
                    states.append(each.state)
                states_tl.update({tlLogic.id: states})
        return states_tl

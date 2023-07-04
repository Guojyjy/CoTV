import time
from lxml import etree

import sumolib
from utils.file_processing import make_xml, print_xml


class SumoScenario:
    def __init__(self, config):
        self.sumo = None
        self.net_file_path = config.get('net')
        self.rou_file_path = config.get('rou')
        self.cfg_file_path = config.get('cfg')
        self.vtype_file_path = config.get('vtype')
        self.phases_all_tls = {}
        self.lanes = []
        self.tls = []

    def get_lane_length(self, lane_id):
        return self.sumo.lane.getLength(lane_id)

    def max_length(self):
        return max(self.get_lane_length(each) for each in self.specify_lanes())

    def max_speed_lane(self, lane_id):
        return self.sumo.lane.getMaxSpeed(lane_id)

    def max_speed(self):
        return max(self.max_speed_lane(each) for each in self.specify_lanes())

    def specify_lanes(self):
        """Map edges to the list of its lanes

        Return
        ------
        edges: dict
            {tl_id: [states]}
        """
        if not self.lanes:
            for edge in sumolib.output.parse(self.net_file_path, ['edge']):
                if not edge.function:  # function="internal"
                    for lane in edge['lane']:
                        self.lanes.append(lane.id)
        return self.lanes

    def node_mapping(self, tl_chosen=None):
        """
        Map the TL with the incoming edges and the outgoing edges

        Returns
        mapping_inc = dict(tl_id: inc_edges)
        num_inc_edges_max, int, maximum number of inc edges
        mapping_out = dict(tl_id: out_edges)
        num_out_edges_max, int, maximum number of out edges
        -------

        """
        mapping_inc = {}
        num_inc_edges_max = 0
        for junction in sumolib.output.parse(self.net_file_path, ['junction']):
            if junction.type == 'traffic_light':
                inc_edges = junction.incLanes.split(' ')
                mapping_inc.update({junction.id: inc_edges})
                num_inc_edges_max = (len(inc_edges), num_inc_edges_max)[num_inc_edges_max > len(inc_edges)]

        mapping_out = {each: [] for each in mapping_inc.keys()}
        num_out_edges_max = 0
        for tl_id, inc_edges in mapping_inc.items():
            out_edges = []
            for connection in sumolib.output.parse(self.net_file_path, ['connection']):
                if connection.attr_from + '_' + connection.fromLane in inc_edges:
                    out_edges.append(connection.to + '_' + connection.toLane)
            mapping_out[tl_id] = list(set(out_edges))
            num_out_edges_max = (len(mapping_out[tl_id]), num_out_edges_max)[
                num_out_edges_max > len(mapping_out[tl_id])]
        # return sorted(mapping.items(), key=lambda x: x[0])
        return mapping_inc, num_inc_edges_max, mapping_out, num_out_edges_max

    def get_tls(self):
        if not self.phases_all_tls and not self.tls:
            for tlLogic in sumolib.output.parse(self.net_file_path, ['tlLogic']):
                self.tls.append(tlLogic.id)
        elif self.phases_all_tls and not self.tls:
            self.tls = self.phases_all_tls.keys()
        return self.tls

    def get_phases_all_tls(self):
        """Map traffic light to the list of its signal phases

        Return
        ------
        phases_tl: dict
            {tl_id: [phases]}
        """
        if self.phases_all_tls == {}:
            for tlLogic in sumolib.output.parse(self.net_file_path, ['tlLogic']):
                phases = []
                for each in tlLogic['phase']:
                    phases.append(each.state)
                self.phases_all_tls.update({tlLogic.id: phases})
        return self.phases_all_tls

    def get_max_accel_vtype(self, vtype_id):
        for vType in sumolib.output.parse(self.vtype_file_path, ['vType']):
            if vType.id == "HDC":
                return float(vType.accel)

    def generate_sumo_cfg(self):
        sumo_cfg = make_xml('configuration', 'http://sumo.dlr.de/xsd/sumoConfiguration.xsd')

        input_content = etree.Element("input")
        input_content.append(etree.Element("net-file", value=self.net_file_path))
        input_content.append(etree.Element("route-files", value=self.rou_file_path))
        sumo_cfg.append(input_content)

        time_content = etree.Element("time")
        time_content.append(etree.Element("begin", value=repr(0)))
        sumo_cfg.append(time_content)

        file_path = self.net_file_path.split('/')[:-1]
        net_name = self.net_file_path.split('/')[-1].split('.')[0]
        print_xml(sumo_cfg, file_path + net_name + '.sumocfg')
        return file_path + net_name + '.sumocfg'

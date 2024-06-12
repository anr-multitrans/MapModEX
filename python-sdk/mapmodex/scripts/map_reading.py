###     Based on data processing code from the official MapTRv2 code available under MIT License
###     Original code can be found at https://github.com/hustvl/MapTR/blob/maptrv2/tools/maptrv2

import sys
import numpy as np
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw

from shapely.geometry import MultiPolygon, Polygon, LineString

from .trajectory import get_nuscenes_trajectory, add_tra_to_vecmap
from .vectorization import get_vect_map

from ..utils import *


class GetMapLayerGeom(object):
    def __init__(self,
                 map_explorer,
                 patch_box=(0,0,60,30),
                 patch_angle=0,
                 avm=None,
                 ego_SE3_city=None,
                 map_classes=['boundary', 'divider', 'ped_crossing',
                              'centerline', 'lane', 'agent'],  # the layers needed
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment'],  # , 'lane'],
                 centerline_classes=['lane_connector', 'lane']):
        super().__init__()
        self.map_explorer = map_explorer
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.polygon_classes = contour_classes + \
            centerline_classes + ped_crossing_classes
        self.patch_box = patch_box
        self.patch_angle = patch_angle
        self.delete = False
        self.avm = avm
        self.ego_SE3_city = ego_SE3_city

        # self.delete_record = delet_record(self.map_explorer, self.pertu_nusc_infos)

    def generate_nearby_centerlines(self, ls_dict):
        from av2.map.map_primitives import Polyline
        for key, value in ls_dict.items():
            value['centerline'] = Polyline.from_array(
                self.avm.get_lane_segment_centerline(key).round(3))
        pts_G = nx.DiGraph()
        junction_pts_list = []
        tmp = ls_dict
        for key, value in tmp.items():
            centerline_geom = LineString(value['centerline'].xyz)
            centerline_pts = np.array(centerline_geom.coords).round(3)
            start_pt = centerline_pts[0]
            end_pt = centerline_pts[-1]
            for idx, pts in enumerate(centerline_pts[:-1]):
                pts_G.add_edge(tuple(centerline_pts[idx]), tuple(
                    centerline_pts[idx+1]))
            valid_incoming_num = 0
            for idx, pred in enumerate(value['predecessors']):
                if pred in tmp.keys():
                    valid_incoming_num += 1
                    pred_geom = LineString(tmp[pred]['centerline'].xyz)
                    pred_pt = np.array(pred_geom.coords).round(3)[-1]
                    pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
            if valid_incoming_num > 1:
                junction_pts_list.append(tuple(start_pt))
            valid_outgoing_num = 0
            for idx, succ in enumerate(value['successors']):
                if succ in tmp.keys():
                    valid_outgoing_num += 1
                    succ_geom = LineString(tmp[succ]['centerline'].xyz)
                    succ_pt = np.array(succ_geom.coords).round(3)[0]
                    pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
            if valid_outgoing_num > 1:
                junction_pts_list.append(tuple(end_pt))
        roots = (v for v, d in pts_G.in_degree() if d == 0)
        leaves = [v for v, d in pts_G.out_degree() if d == 0]
        all_paths = []
        for root in roots:
            paths = nx.all_simple_paths(pts_G, root, leaves)
            all_paths.extend(paths)

        final_centerline_paths = []
        for path in all_paths:
            merged_line = LineString(path)
            merged_line = merged_line.simplify(0.2, preserve_topology=True)
            final_centerline_paths.append(merged_line)

        local_centerline_paths = final_centerline_paths
        return local_centerline_paths

    def generate_nearby_dividers(self, ls_dict):
        left_lane_dict = {}
        right_lane_dict = {}
        for key, value in ls_dict['lane'].items():
            if not value['ls'].is_intersection:
                if value['ls'].left_neighbor_id is not None:
                    left_lane_dict[key] = dict(
                        polyline=value['ls'].left_lane_boundary,
                        predecessors=value['ls'].predecessors,
                        successors=value['ls'].successors,
                        left_neighbor_id=value['ls'].left_neighbor_id,
                    )
                    value['left_lane_divider'] = LineString(
                        value['ls'].left_lane_boundary.xyz)
                if value['ls'].right_neighbor_id is not None:
                    right_lane_dict[key] = dict(
                        polyline=value['ls'].right_lane_boundary,
                        predecessors=value['ls'].predecessors,
                        successors=value['ls'].successors,
                        right_neighbor_id=value['ls'].right_neighbor_id,
                    )
                    value['right_lane_divider'] = LineString(
                        value['ls'].right_lane_boundary.xyz)

        for key, value in left_lane_dict.items():
            if value['left_neighbor_id'] in right_lane_dict.keys():
                del right_lane_dict[value['left_neighbor_id']]

        for key, value in right_lane_dict.items():
            if value['right_neighbor_id'] in left_lane_dict.keys():
                del left_lane_dict[value['right_neighbor_id']]

        for key, value in left_lane_dict.items():
            value['centerline'] = value['polyline']

        for key, value in right_lane_dict.items():
            value['centerline'] = value['polyline']

        left_paths = get_path(left_lane_dict)
        right_paths = get_path(right_lane_dict)
        local_divider = left_paths + right_paths

        return local_divider, ls_dict

    def extract_local_divider(self, ls_dict):
        nearby_dividers, ls_dict = self.generate_nearby_dividers(ls_dict)
        patch = self.map_explorer.get_patch_coord(self.patch_box, self.patch_angle)

        line_list_dic = {}
        for line in nearby_dividers:
            if line.is_empty:  # Skip lines without nodes.
                continue
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        if single_line.is_empty:
                            continue
                        single_line = proc_line(single_line, self.ego_SE3_city)
                        line_dic = {}
                        line_dic['token'] = token_generator()
                        line_dic['geom'] = single_line
                        line_list_dic[line_dic['token']] = line_dic
                else:
                    line = proc_line(line, self.ego_SE3_city)
                    line_dic = {}
                    line_dic['token'] = token_generator()
                    line_dic['geom'] = line
                    line_dic['from'] = 'lane_segments'
                    line_list_dic[line_dic['token']] = line_dic

        ls_dict['divider'] = line_list_dic
        return ls_dict

    def extract_local_ped_crossing(self):
        patch = self.map_explorer.get_patch_coord(self.patch_box, self.patch_angle)

        polygon_list_dic = {}
        for pc in self.avm.get_scenario_ped_crossings():
            exterior_coords = pc.polygon
            interiors = []
            polygon = Polygon(exterior_coords, interiors)
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    if new_polygon.geom_type == 'Polygon':
                        if not new_polygon.is_valid:
                            continue
                        new_polygon = proc_polygon(new_polygon, self.ego_SE3_city)
                        if not new_polygon.is_valid:
                            continue
                    elif new_polygon.geom_type == 'MultiPolygon':
                        polygons = []
                        for single_polygon in new_polygon.geoms:
                            if not single_polygon.is_valid or single_polygon.is_empty:
                                continue
                            new_single_polygon = proc_polygon(
                                single_polygon, self.ego_SE3_city)
                            if not new_single_polygon.is_valid:
                                continue
                            polygons.append(new_single_polygon)
                        if len(polygons) == 0:
                            continue
                        new_polygon = MultiPolygon(polygons)
                        if not new_polygon.is_valid:
                            continue
                    else:
                        raise ValueError(
                            '{} is not valid'.format(new_polygon.geom_type))

                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])

                    polygon_dic = {}
                    polygon_dic['token'] = str(pc.id)
                    polygon_dic['geom'] = new_polygon
                    polygon_dic['from'] = 'pedestrian_crossings'
                    polygon_list_dic[str(pc.id)] = polygon_dic

        return polygon_list_dic

    def extract_local_boundary(self):
        boundary_list = []
        patch = self.map_explorer.get_patch_coord(self.patch_box, self.patch_angle)
        for da in self.avm.get_scenario_vector_drivable_areas():
            boundary_list.append(da.xyz)

        polygon_list_dic = {}
        for da in boundary_list:
            exterior_coords = da
            interiors = []
            polygon = Polygon(exterior_coords, interiors)
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    if polygon.geom_type == 'Polygon':
                        if not polygon.is_valid:
                            continue
                        polygon = proc_polygon(polygon, self.ego_SE3_city)
                        if not polygon.is_valid:
                            continue
                    elif polygon.geom_type == 'MultiPolygon':
                        polygons = []
                        for single_polygon in polygon.geoms:
                            if not single_polygon.is_valid or single_polygon.is_empty:
                                continue
                            new_single_polygon = proc_polygon(
                                single_polygon, self.ego_SE3_city)
                            if not new_single_polygon.is_valid:
                                continue
                            polygons.append(new_single_polygon)
                        if len(polygons) == 0:
                            continue
                        polygon = MultiPolygon(polygons)
                        if not polygon.is_valid:
                            continue
                    else:
                        raise ValueError(
                            '{} is not valid'.format(polygon.geom_type))

                    if polygon.geom_type == 'Polygon':
                        polygon = MultiPolygon([polygon])

                    polygon_dic = {}
                    polygon_dic['token'] = token_generator
                    polygon_dic['geom'] = polygon
                    polygon_dic['from'] = 'drivable_areas'
                    polygon_list_dic[polygon_dic['token']] = polygon_dic

        return polygon_list_dic

    def extract_local_lane(self):
        patch = self.map_explorer.get_patch_coord(self.patch_box, self.patch_angle)
        scene_ls_list = self.avm.get_scenario_lane_segments()
        scene_ls_dict = dict()
        for ls in scene_ls_list:
            scene_ls_dict[ls.id] = dict(
                ls=ls,
                polygon=Polygon(ls.polygon_boundary),
                predecessors=ls.predecessors,
                successors=ls.successors
            )
        ls_dict = dict()
        for key, value in scene_ls_dict.items():
            polygon = value['polygon']
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    ls_dict[key] = value
                    ls_dict[key]['token'] = key
                    ls_dict[key]['geom'] = value['polygon']
                    ls_dict[key]['from'] = 'lane_segments'

        return ls_dict

    def extract_local_centerline(self, ls_dict):
        nearby_centerlines = self.generate_nearby_centerlines(ls_dict['lane'])

        patch = self.map_explorer.get_patch_coord(self.patch_box, self.patch_angle)
        line_list_dic = {}
        for line in nearby_centerlines:
            if line.is_empty:  # Skip lines without nodes.
                continue
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        if single_line.is_empty:
                            continue
                        single_line = proc_line(single_line, self.ego_SE3_city)
                        line_dic = {}
                        line_dic['token'] = token_generator()
                        line_dic['geom'] = single_line

                        line_dic, ls_dict = get_centerline_info(line_dic, ls_dict)

                        line_list_dic[line_dic['token']] = line_dic
                else:
                    line = proc_line(line, self.ego_SE3_city)
                    line_dic = {}
                    line_dic['token'] = token_generator()
                    line_dic['geom'] = line
                    line_dic['from'] = 'lane_segments'

                    line_dic, ls_dict = get_centerline_info(line_dic, ls_dict)

                    line_list_dic[line_dic['token']] = line_dic

        ls_dict['centerline'] = line_list_dic

        return ls_dict
        
    def _get_centerline(self, lane_dict) -> dict:
        centerline_dict = {}
        for lane in lane_dict.values():
            centerline = list(self.map_explorer.map_api.discretize_lanes(
                [lane['token']], 0.5).values())[0]
            centerline = LineString(
                np.array(centerline)[:, :2].round(3))
            if centerline.is_empty:
                continue

            centerline = \
                to_patch_coord(centerline, self.patch_angle, self.patch_box[0], self.patch_box[1])
            record_dict = dict(token=lane['token'],
                                centerline=centerline,
                                incoming_tokens=self.map_explorer.map_api.get_incoming_lane_ids(
                lane['token']),
                outgoing_tokens=self.map_explorer.map_api.get_outgoing_lane_ids(
                    lane['token']),
            )

            centerline_dict.update({lane['token']: record_dict})
        
        return centerline_dict

    def centerline_geoms_to_instances(self, geoms_dict):
        centerline_geoms_list = union_centerline(geoms_dict)
        return one_type_line_geom_to_instances(centerline_geoms_list)

    def get_centerline_line(self, geoms_dict):
        centerline_list = self.centerline_geoms_to_instances(
            geoms_dict['centerline'])
        
        centerline_dics = {}
        for line in centerline_list:
            centerline_dict = {}
            centerline_dict['token'] = token_generator()
            centerline_dict['geom'] = line

            centerline_dict, geoms_dict = get_centerline_info(centerline_dict, geoms_dict)

            centerline_dics[centerline_dict['token']] = centerline_dict

        
        geoms_dict['centerline'], geoms_dict = delete_duplicate_centerline(centerline_dics, geoms_dict)
            
        return geoms_dict

    def get_polyline(self, layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_x = self.patch_box[0]
        patch_y = self.patch_box[1]

        patch = self.map_explorer.get_patch_coord(self.patch_box, self.patch_angle)

        line_list_dic = {}
        records = getattr(self.map_explorer.map_api, layer_name)
        for ind, record in enumerate(records):
            line = self.map_explorer.map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                line = to_patch_coord(line, self.patch_angle, patch_x, patch_y)
                line_dic = {}
                line_dic.update(record)
                line_dic['geom'] = line
                if 'from' not in record:
                    line_dic['from'] = layer_name
                line_list_dic[line_dic['token']] = line_dic

                if self.delete:
                    self.delete_record.delete_layer_record(
                        layer_name, record["token"])

        return line_list_dic

    def get_map_geom(self, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_polyline(layer_name)
            elif layer_name in self.polygon_classes:
                geoms = self.get_polygone(layer_name)

            map_geom.update(geoms)

        return map_geom

    def _valid_polygon(self, polygon, record, layer_name, polygon_list_dic):
        patch = self.map_explorer.get_patch_coord(self.patch_box, self.patch_angle)
        
        patch_x = self.patch_box[0]
        patch_y = self.patch_box[1]

        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                polygon = to_patch_coord(
                    polygon, self.patch_angle, patch_x, patch_y)
                if polygon.geom_type == 'Polygon':
                    polygon = MultiPolygon([polygon])
                polygon_dic = {}
                polygon_dic.update(record)
                polygon_dic['geom'] = polygon
                if 'from' not in record:
                    polygon_dic['from'] = layer_name
                polygon_list_dic[polygon_dic['token']] = polygon_dic

                if self.delete:
                    self.delete_record.delete_layer_record(
                        layer_name, record["token"])
                    
        return polygon_list_dic

    def get_polygone(self, layer_name):
        if layer_name not in self.map_explorer.map_api.lookup_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        records = getattr(self.map_explorer.map_api, layer_name)

        polygon_list_dic = {}
        if layer_name == 'drivable_area':
            for ind, record in enumerate(records):
                polygons = [self.map_explorer.map_api.extract_polygon(
                    polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    polygon_list_dic = self._valid_polygon(polygon, record, layer_name, polygon_list_dic)
        else:
            for ind, record in enumerate(records):
                polygon = self.map_explorer.map_api.extract_polygon(record['polygon_token'])
                polygon_list_dic = self._valid_polygon(polygon, record, layer_name, polygon_list_dic)

        return polygon_list_dic

    def add_isolated_connector(self, map_geom_org_dic):
        if len(map_geom_org_dic['boundary']):
            for lane_dic in map_geom_org_dic['lane'].values():
                if lane_dic['from'] == 'lane_connector':
                    check_intersection = 0
                    for bd in map_geom_org_dic['boundary'].values():
                        if lane_dic['geom'].intersection(bd['geom']):
                            check_intersection = 1
                            break
                    
                    if not check_intersection:
                        lane_dic['from'] == ['lane_connector_i']
                        
        return map_geom_org_dic
class get_vec_map():
    """transfer original data from map database to vector map, perturbation is optional."""
    def __init__(self, info, nusc_maps, map_explorer, out_path, e2g_translation=None, e2g_rotation=None, pc_range=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
                 out_type='json', nusc=None, avm=None, vis=True, mme=False):
        """ initialization

        Args:
            info (dict): Information from the original map
            nusc_maps (NuScenesMap): Dataset class in the nuScenes map dataset. 
            map_explorer (NuScenesMapExplorer): Dataset class in the nuScenes map dataset explorer.
            out_path (str): output path
            e2g_translation (_type_, optional): The conversion relationship between the agent's current coordinates and the global coordinates. Defaults to None.
            e2g_rotation (_type_, optional): The conversion relationship between the agent's current angle and the global angle. Defaults to None.
            pc_range (list, optional): patch box size(3D). Defaults to [-30.0, -15.0, -5.0, 30.0, 15.0, 3.0].
            out_type (str, optional): output type. 'pkl' is the data used for model training, and 'json' is the map data. Defaults to 'json'.
            nusc (NuScenesd, optional): Dataset class in the nuScenes dataset. Defaults to None.
            avm (ArgoverseStaticMap, optional): Dataset class in the Argoverse 2 dataset. Defaults to None.
            vis (bool, optional): visulization. Defaults to True.
            mme (bool, optional): map dataset from MapModEX. Defaults to False.
        """
        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.map_explorer = map_explorer
        self.info = info
        self.vis_switch = vis
        self.e2g_translation = e2g_translation
        self.e2g_rotation = e2g_rotation
        self.pc_range = pc_range
        self.avm = avm
        self.output_type = out_type
        self.mme = mme

        self.vec_classes = ['boundary', 'lane', 'divider',
                            'ped_crossing', 'centerline', 'agent']
        
        # set the final plot and saved layers
        self.info['order'] = ['boundary', 'divider', 'ped_crossing', 'centerline'] #, 'agent']
        self._get_patch_info()

        vis_path = os.path.join(out_path, 'visualization')
        self.visual = RenderMap(self.info, vis_path, vis)

        # map info setting
        if out_type == 'json':
            self.map_path = os.path.join(out_path, 'MME_map')
        else:
            self.map_path = None

    def _get_patch_info(self):
        patch_h = self.pc_range[3]-self.pc_range[0]
        patch_w = self.pc_range[4]-self.pc_range[1]
        self.patch_size = (patch_h, patch_w)

        if self.e2g_translation is not None:
            map_pose = self.e2g_translation[:2]
            self.patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        else:
            self.patch_box = (0, 0, self.patch_size[0], self.patch_size[1])

        if self.avm is not None:
            rotation = Quaternion._from_matrix(self.e2g_rotation)
            from av2.geometry.se3 import SE3
            city_SE2_ego = SE3(self.e2g_rotation, self.e2g_translation)
            self.ego_SE3_city = city_SE2_ego.inverse()
        else:
            if self.e2g_rotation is not None:
                rotation = Quaternion(self.e2g_rotation)
            self.ego_SE3_city = None
        if self.e2g_rotation is not None:
            self.patch_angle = quaternion_yaw(rotation) / np.pi * 180
        else:
            self.patch_angle = 0
        # map class setting
        self.get_geom = GetMapLayerGeom(self.map_explorer, self.patch_box, self.patch_angle, self.avm, self.ego_SE3_city, self.vec_classes)

    def _get_map_nuscenes(self):
        map_geom_org_dic = {}
        for vec_class in self.vec_classes:
            if vec_class == 'boundary':  # road_segment
                map_geom_org_dic[vec_class] = self.get_geom.get_map_geom(['road_segment'])
            elif vec_class == 'lane':  # road_divider, lane_divider
                map_geom_org_dic[vec_class] = self.get_geom.get_map_geom(['lane', 'lane_connector'])
            elif vec_class == 'divider':  # road_divider, lane_divider
                map_geom_org_dic[vec_class] = self.get_geom.get_map_geom(['road_divider', 'lane_divider']) #, 'lane_divider'])
            elif vec_class == 'ped_crossing':  # oed_crossing
                map_geom_org_dic[vec_class] = self.get_geom.get_map_geom(['ped_crossing'])
            elif vec_class == 'centerline':  # lane, connector
                map_geom_org_dic[vec_class] = self.get_geom._get_centerline(map_geom_org_dic['lane'])
                map_geom_org_dic = self.get_geom.get_centerline_line(map_geom_org_dic)
            elif vec_class == 'agent':
                agents_trajectory = get_nuscenes_trajectory(self.nusc, self.info['token'], ['vehicle'], type='geom')
                map_geom_org_dic[vec_class] = add_tra_to_vecmap(agents_trajectory, self.map_explorer, self.patch_box, self.patch_angle)
                map_geom_org_dic = get_agent_info(map_geom_org_dic)
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')
        
        map_geom_org_dic = self.get_geom.add_isolated_connector(map_geom_org_dic)
        
        return map_geom_org_dic
    
    def _get_map_av2(self):
        map_geom_org_dic = {}
        for vec_class in self.vec_classes:
            if vec_class == 'ped_crossing':  # oed_crossing
                map_geom_org_dic[vec_class] = self.get_geom.extract_local_ped_crossing()
            elif vec_class == 'boundary':  # road_segment
                map_geom_org_dic[vec_class] = self.get_geom.extract_local_boundary()
            elif vec_class == 'lane':  # lane, connector
                map_geom_org_dic[vec_class] = self.get_geom.extract_local_lane()
            elif vec_class == 'centerline':  # lane, connector
                map_geom_org_dic = self.get_geom.extract_local_centerline(map_geom_org_dic)
            elif vec_class == 'divider':  # road_divider, lane_divider
                map_geom_org_dic = self.get_geom.extract_local_divider(map_geom_org_dic)
            elif vec_class == 'agent':
                map_geom_org_dic[vec_class] = {}  # TODO
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')
        return map_geom_org_dic
    
    def _get_map_mme(self):
        self.get_geom = GetMapLayerGeom(self.map_explorer, self.patch_box, self.patch_angle, map_classes=self.vec_classes,
                                        line_classes=['divider', 'centerline', 'agent', 'boundary'])
        map_geom_org_dic = {}
        for vec_class in self.vec_classes:
            if vec_class in ['boundary', 'lane', 'divider', 'ped_crossing', 'centerline', 'agent']:
                map_geom_org_dic[vec_class] = self.get_geom.get_map_geom([vec_class])
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')
            
            return map_geom_org_dic
    
    def get_map_ann(self, pertube_vers):
        """get map layers and transfer them to vector, perturbation is optional

        Args:
            pertube_vers (list): perturbed versions, each version should be a dict with parameter_names and parameter_values.

        Returns:
            dict: information include vector map layers
        """
        # get geom for layers and transfer line string geom to instance
        if self.info['dataset'] == 'nuscenes':
            self.info['map_geom_org_dic'] = self._get_map_nuscenes()
        elif self.info['dataset'] == 'av2':
            self.info['map_geom_org_dic'] = self._get_map_av2()
        elif self.info['dataset'] == 'mme':
            self.info['map_geom_org_dic'] = self._get_map_mme()
        else:
            sys.exit('wrong dataset')
        
        self.info = get_vect_map(self.nusc_maps, self.map_explorer, pertube_vers, self.info, self.visual, self.vis_switch, self.map_path,
                                 self.output_type, self.mme)
        
        return self.info

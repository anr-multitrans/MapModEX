###     Based on data processing code from the official MapTRv2 code available under MIT License
###     Original code can be found at https://github.com/hustvl/MapTR/blob/maptrv2/tools/maptrv2

import copy
import numpy as np

from shapely import ops
from shapely.geometry import MultiLineString, MultiPolygon, box, LineString

from .peturbation import MapTransform

# from utils.visualization import RenderMap
from ..utils import *


class VectorizedMap(object):
    """transform geometry map layers to vector"""
    def __init__(self,
                 nusc_map,
                 map_explorer,
                 patch_box=(0,0,60,30),
                 patch_angle=0,
                 avm=None,
                 ego_SE3_city=None,
                 map_classes=['boundary', 'divider', 'ped_crossing',
                              'centerline', 'lane', 'agent'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment'],
                 centerline_classes=['lane_connector', 'lane'],
                 mme=False):
        """ initialization

        Args:
            nusc_map (NuScenesMap): Dataset class in the nuScenes map dataset. 
            map_explorer (NuScenesMapExplorer): Dataset class in the nuScenes map dataset explorer.
            patch_box (tuple, optional): patch box[x,y,hight,width]. Defaults to (0,0,60,30).
            patch_angle (int, optional): patch angle. Defaults to 0.
            avm (ArgoverseStaticMap, optional): Dataset class in the Argoverse 2 dataset. Defaults to None.
            ego_SE3_city (optional): Used for converting AV2's current seat position and the global position. Defaults to None.
            map_classes (list, optional): Unified map layer types for perturbations. Defaults to ['boundary', 'divider', 'ped_crossing', 'centerline', 'lane', 'agent'].
            ped_crossing_classes (list, optional): map layer types belonging to the crosswalk. Defaults to ['ped_crossing'].
            contour_classes (list, optional): map layer types that belong to boundaries. Defaults to ['road_segment'].
            centerline_classes (list, optional): The type of map layer used to generate the path. Defaults to ['lane_connector', 'lane'].
            mme (bool, optional): map dataset from MapModEX. Defaults to False.
        """
        super().__init__()
        self.nusc_map = nusc_map
        self.map_explorer = map_explorer
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes + \
            centerline_classes + ped_crossing_classes
        self.centerline_classes = centerline_classes
        self.patch_box = patch_box
        self.patch_angle = patch_angle
        self.delete = False
        self.avm = avm
        self.ego_SE3_city = ego_SE3_city
        self.mme = mme

        # delete_record = delet_record(
        #     map_explorer, pertu_nusc_infos)

    def _init_pertu_nusc_infos(self, empty=True) -> None:
        if empty:
            self.pertu_nusc_infos = {"polygon": [],
                                     "line": [],
                                     "node": [],
                                     "boundary": [],
                                     "divider": [],
                                     "ped_crossing": [],
                                     "drivable_area": [],
                                     "road_segment": [],
                                     "road_block": [],
                                     "lane": [],
                                     "walkway": [],
                                     "stop_line": [],
                                     "carpark_area": [],
                                     "road_divider": [],
                                     "lane_divider": [],
                                     "traffic_light": [],
                                     "centerline": [],
                                     "agent": [],
                                     "canvas_edge": self.nusc_map.canvas_edge,
                                     "version": self.nusc_map.version,
                                     "arcline_path_3": [],
                                     "connectivity": [],
                                     "lane_connector": []}  # nusc_maps.json
        else:
            self.pertu_nusc_infos = {"polygon": copy.deepcopy(self.nusc_map.polygon),
                                     "line": copy.deepcopy(self.nusc_map.line),
                                     "node": copy.deepcopy(self.nusc_map.node),
                                     "road_segment": copy.deepcopy(self.nusc_map.road_segment),
                                     "lane": copy.deepcopy(self.nusc_map.lane),
                                     "boundary": [],
                                     "road_divider": copy.deepcopy(self.nusc_map.road_divider),
                                     "lane_divider": copy.deepcopy(self.nusc_map.lane_divider),
                                     "divider": [],
                                     "ped_crossing": copy.deepcopy(self.nusc_map.ped_crossing),
                                     "centerline": [],
                                     "agent": [],
                                     "canvas_edge": self.nusc_map.canvas_edge,
                                     "version": self.nusc_map.version}  # nusc_maps.json
            self.delete = True

    def gen_vectorized_samples(self, map_geom_dic):
        """transfer map layers to the instance that can be visualized

        Args:
            map_geom_dic (dict): geometry map layer

        Raises:
            ValueError: if there is a wrong layer type

        Returns:
            dict: geometry map instance
        """
        map_ins_org_dict = {}
        
        map_ins_org_dict = make_dict(self.vec_classes)
        
        for vec_class in ['centerline', 'agent']:
            if vec_class in map_geom_dic:
                for v in map_geom_dic[vec_class].values():
                    map_ins_org_dict[vec_class].append(v['geom'])
        
        # transfer non LineString geom to an instance
        for vec_class in map_geom_dic.keys():
            if vec_class in ['centerline', 'agent']:
                continue
            elif vec_class == 'boundary':
                map_ins_org_dict[vec_class] = self._poly_geoms_to_instances(map_geom_dic) # merge boundary and lanes
            elif vec_class == 'divider':
                # take from 'divider' and 'lane', merge overlapped and delete duplicated
                map_ins_org_dict[vec_class] = self._line_geoms_to_instances(map_geom_dic) 
            elif vec_class == 'ped_crossing':
                continue
            elif vec_class == 'lane':
                for v in map_geom_dic[vec_class].values():
                    if v['from'] == 'lane_connector':
                        continue
                    map_ins_org_dict[vec_class].append(v['geom'])
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')

        # merge overlapped or connected ped_crossing to one
        if 'ped_crossing' in map_geom_dic:
            map_ins_org_dict['ped_crossing'] = self.ped_poly_geoms_to_instances(map_geom_dic['ped_crossing'], map_ins_org_dict)

        if 'divider' in map_ins_org_dict.keys():
            new_dividers = []
            for line in map_ins_org_dict['divider']:
                divider_check = 1
                for boundary in map_ins_org_dict['boundary']:
                    if line.intersection(boundary):
                        divider_check = 0
                        break

                if divider_check:
                    check_intersection = 0
                    for ped in map_ins_org_dict['ped_crossing']:
                        if line.intersection(ped):
                            check_intersection = 1
                            new_dividers += keep_non_intersecting_parts(line, ped)
                            break
                    
                    if not check_intersection:
                        new_dividers.append(line)

            map_ins_org_dict['divider'] = new_dividers
        
        return map_ins_org_dict
        
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


    def get_centerline_line(self, geoms_dict):
        centerline_list = self.centerline_geoms_to_instances(
            geoms_dict['centerline'])
        
        centerline_dics = {}
        for line in centerline_list:
            centerline_dict = {}
            centerline_dict['token'] = token_generator()
            centerline_dict['geom'] = line

            centerline_dict, geoms_dict = get_centerline_info(
                line, centerline_dict, geoms_dict)

            centerline_dics[centerline_dict['token']] = centerline_dict

        geoms_dict['centerline'] = centerline_dics

        return geoms_dict

    def _line_geoms_to_instances(self, geom_dict):
        line_geom_list = {}

        if 'divider' in geom_dict.keys():
            for k, divider_dic in geom_dict['divider'].items():
                if divider_dic['from'] == 'road_divider':
                    # line_geom_list.update(divider_dic)
                    line_geom_list[k] = divider_dic
            
        if 'lane' in geom_dict.keys():
            for lane_dic in geom_dict['lane'].values():
                if lane_dic['from'] == 'lane': 
                    for div_name in ['left_lane_divider_token', 'right_lane_divider_token']:
                        if div_name in lane_dic:
                            if lane_dic[div_name] in geom_dict['divider']:
                                line_geom_list[lane_dic[div_name]] = geom_dict['divider'][lane_dic[div_name]]

        line_instances = [divider['geom'] for divider in line_geom_list.values()]
        new_lans = [lane_dic['geom'] for lane_dic in geom_dict['lane'].values() if lane_dic['from'] in ['centerline']]
        
        if line_instances and new_lans:
            for lane in new_lans:
                line_instances_temp = []
                for divider in line_instances:
                    if not divider.is_empty:
                        int_divider = interpolate(divider)
                        line_instances_temp += keep_non_intersecting_parts(int_divider, lane, True)
                line_instances = copy.deepcopy(line_instances_temp)

        return line_instances

    def ped_poly_geoms_to_instances(self, ped_geom, map_ins_dict=[]):
        ped = []

        for ped_dic in ped_geom.values():
            if ped_dic['geom'].geom_type in ['Polygon', 'MultiPolygon']:
                if ped_dic['from'] == 'new':
                    for bon in map_ins_dict['boundary']:
                        if bon.intersection(ped_dic['geom']):
                            if_inter_divider = False
                            for div in map_ins_dict['divider']:
                                if div.intersection(ped_dic['geom']):
                                    if_inter_divider = True
                                    break
                            if not if_inter_divider:
                                ped.append(ped_dic['geom'])
                                break
                else:
                    ped.append(ped_dic['geom'])

        union_segments = ops.unary_union(ped)
        
        # max_x = self.patch_box[3] / 2
        # max_y = self.patch_box[2] / 2
        # local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        max_xy = max(self.patch_box[2:]) / 2
        local_patch = box(-max_xy, -max_xy, max_xy, max_xy)
        
        exteriors = []
        interiors = []
        if union_segments.geom_type == 'Polygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext = LineString(list(ext.coords)[::-1])
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                ext = LineString(list(ext.coords)[::-1])
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return one_type_line_geom_to_instances(results)

    def _poly_geoms_to_instances(self, polygon_geom, mme=False):
        boundary = []
        lane_connector = []
        polygons = []
        
        if not self.mme:
            if 'boundary' in polygon_geom.keys():
                for road_dic in polygon_geom['boundary'].values():
                    boundary.append(road_dic['geom'])

        if 'lane' in polygon_geom.keys():
            for lane_dic in polygon_geom['lane'].values():
                if lane_dic['from'] != 'lane_connector':
                    polygons.append(lane_dic['geom'])
                else:
                    lane_connector.append(lane_dic['geom'])
                    
        if len(boundary) and len(lane_connector):
            union_boundary = ops.unary_union(boundary)
            for lane_con in lane_connector:
                if lane_con.intersection(union_boundary):
                    continue
                else:
                    polygons.append(lane_con)
                    
        polygons += boundary

        union_segments = ops.unary_union(polygons)
        
        # max_x = self.patch_box[3] / 2
        # max_y = self.patch_box[2] / 2
        # local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        max_xy = max(self.patch_box[2:]) / 2
        local_patch = box(-max_xy, -max_xy, max_xy, max_xy)
        
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            # lines = LineString(ext)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            # lines = LineString(inter)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        results = one_type_line_geom_to_instances(results)

        # for ind, d in enumerate(results):
        #     results[ind] = remove_polyline_overlap(results[ind])
        
        delete_boundary = []
        for ind, line in enumerate(results):
            for cl in polygon_geom['centerline'].values():
                if line.intersection(cl['geom']) and line.length < 3.5:
                    delete_boundary.append(ind)
        
        results = delete_elements_by_indices(results, delete_boundary)            
        
        return results


def get_vect_map(nusc_maps, map_explorer, pertube_vers, info, visual, vis_switch=False, map_path='', output_type='json', mme=False):
    """transform organized map layers to vector, perturbat if necessary.

    Args:
        nusc_maps (NuScenesMap): Dataset class in the nuScenes map dataset.
        map_explorer (NuScenesMapExporer): Dataset class in the nuScenes map dataset explorer.
        pertube_vers (list): perturbed versions, each version should be a dict with parameter_names and parameter_values.
        info (dcit): information from the original map 
        vis_switch (bool): visualization switch
        visual (RenderMap): class in map visualization
        map_path (str): output path for saving map data
        output_type (str, optional): output type. 'pkl' is the data used for model training, and 'json' is the map data. Defaults to 'json'.
        mme (bool, optional): map dataset from MapModEX. Defaults to False.

    Returns:
        _type_: _description_
    """
    vector_map = VectorizedMap(nusc_maps, map_explorer, mme=mme)
    map_trans = MapTransform(map_explorer, info['map_geom_org_dic'], vector_map, visual)    

    for map_v in pertube_vers:
        if map_v.pt_name:
            ann_name = 'annotation' + '_' + map_v.pt_name
            map_json_name = 'mme' + '_' + map_v.pt_name

        map_trans.tran_args = map_v
        map_trans.ann_name = ann_name
        
        ## geom level perturbation
        map_trans.perturb_geom_layer(copy.deepcopy(info['map_geom_org_dic']))
        
        ## crop the ready-show vectorized geom by patch_box.
        trans_dic = {}
        map_ins_dict = {}
        geom_dic_copy = copy.deepcopy(map_trans.geom_dict)
        for vec_class, pt_vect_dic in geom_dic_copy.items():
            map_ins_dict[vec_class] = get_geom_in_patch(map_explorer, pt_vect_dic, vector_map.patch_box)
        trans_dic['map_ins_dic_patch'] = map_ins_dict
        
        ## vectetry level pertubation
        map_trans.perturb_vect_map()
        map_vect_pt_dic = copy.deepcopy(map_trans.vect_dict)
            
        if (map_v.int_num and map_v.int_ord == 'after') or (not map_v.int_num and map_v.int_sav):
            map_vect_pt_dic = np_to_geom(map_vect_pt_dic)
            map_vect_pt_dic = geom_to_np(
                map_vect_pt_dic, inter=True, inter_args=map_v.int_num)
            if vis_switch:
                visual.vis_contours(map_vect_pt_dic, ann_name)

        elif map_v.int_num and not map_v.int_sav:
            if vis_switch:
                visual.vis_contours(map_vect_pt_dic, ann_name)
            map_vect_pt_dic = np_to_geom(map_vect_pt_dic)
            trans_ins_np = geom_to_np(map_ins_dict)
            map_vect_pt_dic = geom_to_np(
                map_vect_pt_dic, inter=True, inter_args=trans_ins_np, int_back=True)

        else:  # not map_v.int_num and not map_v.int_sav
            if vis_switch:
                # trans_np_dict_4_vis = np_to_geom(map_vect_pt_dic)
                # trans_np_dict_4_vis = geom_to_np(trans_np_dict_4_vis, inter=True)
                visual.vis_contours(map_vect_pt_dic, ann_name)

        ## Crop the ready-show vectorized numpy array by patch_box.
        map_geom_dict = {}
        for vec_class, pt_geom_dic in map_vect_pt_dic.items():
            map_geom_dict[vec_class] = get_vect_in_patch(pt_geom_dic, vector_map.patch_box)
        
        if output_type == 'json':
            # keep the instance part only in the patch.
            map_geom_dict = {}
            for vec_class, pt_geom_dic in map_trans.geom_dict_for_json.items():
                map_geom_dict[vec_class] = get_geom_in_patch(map_explorer, pt_geom_dic, vector_map.patch_box)
            
            trans_dic['map_geom_dic_patch'] = map_geom_dict
            
            vector_map._init_pertu_nusc_infos()
            trans_dic['pertu_nusc_infos'] = vector_map.pertu_nusc_infos

            info[map_json_name] = vector_to_map_json(trans_dic, info, map_json_name, map_path) #TODO use pt-vect

        info[ann_name] = map_geom_dict
    
    return info



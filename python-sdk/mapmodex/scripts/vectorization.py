import copy
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMapExplorer

from shapely import affinity, ops
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, box, LineString

from .peturbation import MapTransform

# from utils.visualization import RenderMap
from ..utils import *


class VectorizedLocalMap(object):
    def __init__(self,
                 nusc,
                 nusc_map,
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
        self.nusc = nusc
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
        self._init_pertu_nusc_infos()
        
        '''get transformed gt map layers'''
        map_ins_org_dict = {}
        
        map_ins_org_dict = make_dict(self.vec_classes)
        # transfer non linestring geom to instance
        for vec_class in map_geom_dic.keys():
            if vec_class in ['lane', 'centerline', 'agent']:
                # map_ins_org_dict[vec_class] = ped_poly_geoms_to_instances(
                #     map_geom_dic[vec_class])
                for v in map_geom_dic[vec_class].values():
                    map_ins_org_dict[vec_class].append(v['geom'])
            elif vec_class == 'boundary':
                map_ins_org_dict[vec_class] = self.poly_geoms_to_instances(map_geom_dic)   # merge boundary and lanes
            elif vec_class == 'divider':
                map_ins_org_dict[vec_class] = self.line_geoms_to_instances(map_geom_dic) # take from 'divier' and 'lane', merge overlaped and delete duplicated
            elif vec_class == 'ped_crossing':
                continue
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')

        if 'ped_crossing' in map_geom_dic:
            map_ins_org_dict['ped_crossing'] = self.ped_poly_geoms_to_instances(map_geom_dic['ped_crossing'], map_ins_org_dict) # merge overlaped or connected ped_crossing to one

        if 'divider' in map_ins_org_dict.keys():
            new_dividers = []
            for line in map_ins_org_dict['divider']:
                divider_check = 1
                for boundary in map_ins_org_dict['boundary']:
                    if line.intersection(boundary):
                        divider_check = 0
                        break

                if divider_check:
                    new_dividers.append(line)

            map_ins_org_dict['divider'] = new_dividers

        return {'map_ins_org_dict': map_ins_org_dict, 'map_geom_dic': map_geom_dic, 'pertu_nusc_infos': self.pertu_nusc_infos}
        
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

    def line_geoms_to_instances(self, geom_dict):
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
        max_x = self.patch_box[3] / 2
        max_y = self.patch_box[2] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
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

    def poly_geoms_to_instances(self, polygon_geom):
        polygons = []
        
        if 'boundary' in polygon_geom.keys():
            for road_dic in polygon_geom['boundary'].values():
                polygons.append(road_dic['geom'])

        if 'lane' in polygon_geom.keys():
            for lane_dic in polygon_geom['lane'].values():
                if lane_dic['from'] != 'lane_connector':
                    polygons.append(lane_dic['geom'])

        union_segments = ops.unary_union(polygons)
        max_x = self.patch_box[3] / 2
        max_y = self.patch_box[2] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
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
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return one_type_line_geom_to_instances(results)


def get_vect_map(nusc, nusc_maps, map_explorer, pertube_vers, info, vis_switch, visual, map_path):
    vector_map = VectorizedLocalMap(nusc, nusc_maps, map_explorer)
    map_trans = MapTransform(map_explorer)    

    for map_v in pertube_vers:
        if map_v.pt_name:
            ann_name = 'annotation' + '_' + map_v.pt_name
            map_json_name = 'mme' + '_' + map_v.pt_name

        map_trans.tran_args = map_v
        
        map_geom_org_dic = copy.deepcopy(info['map_geom_org_dic'])
        
        ## geom level pertubation
        map_geom_pt_dic = map_trans.pertub_geom(map_geom_org_dic)
        
        trans_dic = vector_map.gen_vectorized_samples(map_geom_pt_dic)
        
        # keep the instance part only in the patch.
        map_ins_dict = {}
        for vec_class, pt_vect_dic in trans_dic['map_ins_org_dict'].items():
            map_ins_dict[vec_class] = get_geom_in_patch(map_explorer, pt_vect_dic, vector_map.patch_box)
        trans_dic['map_ins_dic_patch'] = map_ins_dict
        
        if map_v.int_num and map_v.int_ord == 'before':
            map_vect_dic = geom_to_np(map_ins_dict, inter=True, inter_args=map_v.int_num)
        else:
            map_vect_dic = geom_to_np(map_ins_dict)
        
        ## vectetry level pertubation
        map_vect_pt_dic  = map_trans.perturb_vect(map_vect_dic)
            
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
                trans_np_dict_4_vis = np_to_geom(map_vect_pt_dic)
                trans_np_dict_4_vis = geom_to_np(
                    trans_np_dict_4_vis, inter=True)
                visual.vis_contours(trans_np_dict_4_vis, ann_name)

        if map_path is not None:
            # keep the instance part only in the patch.
            map_geom_dict = {}
            for vec_class, pt_geom_dic in trans_dic['map_geom_dic'].items():
                map_geom_dict[vec_class] = get_geom_in_patch(map_explorer, pt_geom_dic, vector_map.patch_box)
            
            trans_dic['map_geom_dic_patch'] = map_geom_dict
            # trans_dic['map_ins_dic_patch'] = map_ins_dict
            info[map_json_name] = vector_to_map_json(trans_dic, info, map_json_name, map_path) #TODO use pt-vect

        info[ann_name] = map_vect_pt_dic
    
    return info



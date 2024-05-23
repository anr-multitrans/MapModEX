import asyncio
import copy
import sys
import numpy as np
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMapExplorer

from shapely import affinity, ops
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, box, LineString

from .trajectory import get_nuscenes_trajectory, add_tra_to_vecmap
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
        self.map_trans = MapTransform(self.map_explorer)
        self.delete = False
        self.avm = avm
        self.ego_SE3_city = ego_SE3_city

        # self.delete_record = delet_record(
        #     self.map_explorer, self.pertu_nusc_infos)

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

    def gen_vectorized_samples(self, map_geom_org_dic, tran_args=None):
        self._init_pertu_nusc_infos()
        
        '''get transformed gt map layers'''
        map_ins_org_dict = {}
        map_geom_dict = {}
        map_ins_dict = {}

        # geom level pertubation
        if tran_args is not None: 
            if tran_args.del_lan[0]:
                map_geom_org_dic = self.map_trans.del_centerline(map_geom_org_dic, tran_args.del_lan)

            if tran_args.add_lan[0]:
                map_geom_org_dic = self.map_trans.add_centerline(map_geom_org_dic, tran_args.add_lan)
            
            if tran_args.wid_lan[0]:
                map_geom_org_dic = self.map_trans.widden_lane(map_geom_org_dic, tran_args.wid_lan)
                map_geom_org_dic['centerline'] = self._get_centerline(map_geom_org_dic['lane'])
                map_geom_org_dic = self.get_centerline_line(map_geom_org_dic)

            if tran_args.aff_tra_pat[0] or tran_args.rot_pat[0] or tran_args.sca_pat[0] or tran_args.ske_pat[0] or tran_args.shi_pat[0]:
                map_geom_org_dic = self.map_trans.transfor_patch(map_geom_org_dic, tran_args)

        # keep the instance part only in the patch.
        for vec_class in map_geom_org_dic.keys():
            map_geom_dict[vec_class] = get_geom_in_patch(self.map_explorer, map_geom_org_dic[vec_class], [
                                                        0, 0, self.patch_box[2], self.patch_box[3]])
        
        map_ins_org_dict = make_dict(self.vec_classes)
        # transfer non linestring geom to instance
        for vec_class in map_geom_org_dic.keys():
            if vec_class in ['lane', 'centerline', 'agent']:
                # map_ins_org_dict[vec_class] = self.ped_poly_geoms_to_instances(
                #     map_geom_org_dic[vec_class])
                for v in map_geom_org_dic[vec_class].values():
                    map_ins_org_dict[vec_class].append(v['geom'])
            elif vec_class == 'ped_crossing':
                map_ins_org_dict[vec_class] = self.ped_poly_geoms_to_instances(
                    map_geom_org_dic[vec_class]) # merge overlaped or connected ped_crossing to one
            elif vec_class == 'boundary':
                map_ins_org_dict[vec_class] = self.poly_geoms_to_instances(
                    map_geom_org_dic)   # merge boundary and lanes
            elif vec_class == 'divider':
                map_ins_org_dict[vec_class] = self.line_geoms_to_instances(map_geom_org_dic) # take from 'divier' and 'lane', merge overlaped and delete duplicated
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')

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

        # keep the instance part only in the patch.
        for vec_class in map_ins_org_dict.keys():
            map_ins_dict[vec_class] = get_geom_in_patch(self.map_explorer, map_ins_org_dict[vec_class], [
                                                        0, 0, self.patch_box[2], self.patch_box[3]])

        return {'map_ins_dict': map_ins_dict, 'map_geom_dic': map_geom_dict, 'map_ins_org_dict': map_ins_org_dict, 'map_geom_org_dic': map_geom_org_dic, 'pertu_nusc_infos': self.pertu_nusc_infos}

    def gen_vectorized_samples_by_pt_json(self, patch_box=[0, 0, 60, 30], patch_angle=0):
        # get transformed gt map layers
        patch_box = [0, 0, 60, 30]
        patch_angle = 0

        map_ins_dict = {}

        for vec_class in self.vec_classes:
            line_geom = self.get_polyline(patch_box, patch_angle, vec_class)
                            
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

    def generate_nearby_centerlines(self, avm, ls_dict):
        from av2.map.map_primitives import Polyline
        for key, value in ls_dict.items():
            value['centerline'] = Polyline.from_array(
                avm.get_lane_segment_centerline(key).round(3))
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

    def extract_local_divider(self, ego_SE3_city, patch_box, patch_angle, ls_dict):
        nearby_dividers, ls_dict = self.generate_nearby_dividers(ls_dict)
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)

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
                        single_line = proc_line(single_line, ego_SE3_city)
                        line_dic = {}
                        line_dic['token'] = token_generator()
                        line_dic['geom'] = single_line
                        line_list_dic[line_dic['token']] = line_dic
                else:
                    line = proc_line(line, ego_SE3_city)
                    line_dic = {}
                    line_dic['token'] = token_generator()
                    line_dic['geom'] = line
                    line_list_dic[line_dic['token']] = line_dic

        ls_dict['divider'] = line_list_dic
        return ls_dict

    def extract_local_ped_crossing(self, avm, ego_SE3_city, patch_box, patch_angle):
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)

        polygon_list_dic = {}
        for pc in avm.get_scenario_ped_crossings():
            exterior_coords = pc.polygon
            interiors = []
            polygon = Polygon(exterior_coords, interiors)
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    if new_polygon.geom_type == 'Polygon':
                        if not new_polygon.is_valid:
                            continue
                        new_polygon = proc_polygon(new_polygon, ego_SE3_city)
                        if not new_polygon.is_valid:
                            continue
                    elif new_polygon.geom_type == 'MultiPolygon':
                        polygons = []
                        for single_polygon in new_polygon.geoms:
                            if not single_polygon.is_valid or single_polygon.is_empty:
                                continue
                            new_single_polygon = proc_polygon(
                                single_polygon, ego_SE3_city)
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
                    polygon_list_dic[str(pc.id)] = polygon_dic

        return polygon_list_dic

    def extract_local_boundary(self, avm, ego_SE3_city, patch_box, patch_angle):
        boundary_list = []
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        for da in avm.get_scenario_vector_drivable_areas():
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
                        polygon = proc_polygon(polygon, ego_SE3_city)
                        if not polygon.is_valid:
                            continue
                    elif polygon.geom_type == 'MultiPolygon':
                        polygons = []
                        for single_polygon in polygon.geoms:
                            if not single_polygon.is_valid or single_polygon.is_empty:
                                continue
                            new_single_polygon = proc_polygon(
                                single_polygon, ego_SE3_city)
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
                    polygon_list_dic[polygon_dic['token']] = polygon_dic

        return polygon_list_dic

    def extract_local_lane(self, avm, patch_box, patch_angle):
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        scene_ls_list = avm.get_scenario_lane_segments()
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

        return ls_dict

    def extract_local_centerline(self, avm, ego_SE3_city, patch_box, patch_angle, ls_dict):
        nearby_centerlines = self.generate_nearby_centerlines(
            avm, ls_dict['lane'])

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
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
                        single_line = proc_line(single_line, ego_SE3_city)
                        line_dic = {}
                        line_dic['token'] = token_generator()
                        line_dic['geom'] = single_line

                        line_dic, ls_dict = get_centerline_info(
                            single_line, line_dic, ls_dict)

                        line_list_dic[line_dic['token']] = line_dic
                else:
                    line = proc_line(line, ego_SE3_city)
                    line_dic = {}
                    line_dic['token'] = token_generator()
                    line_dic['geom'] = line

                    line_dic, ls_dict = get_centerline_info(
                        line, line_dic, ls_dict)

                    line_list_dic[line_dic['token']] = line_dic

        ls_dict['centerline'] = line_list_dic

        return ls_dict

    def _get_lane_divider(self, lane_dicts):
        for lane_dic in lane_dicts.values():
            if lane_dic['from'] == 'lane':
                lane_record = lane_dic['record']
                for direction in ['left', 'right']:
                    divider_name = direction + '_lane_divider_segments'
                    if len(lane_record[divider_name]):
                        lane_segments = list(set([seg['node_token'] for seg in lane_record[divider_name]]))
                        node_records = getattr(self.map_explorer.map_api, 'node')
                        nodes = []
                        for rec in node_records:
                            if rec['token'] in lane_segments:
                                nodes.append([rec['x'], rec['y']])
                        divider = LineString(nodes)
                        divider = to_patch_coord(divider, self.patch_angle, self.patch_box[0], self.patch_box[1])
                        lane_dic[direction + '_lane_divider'] = divider
        
        return lane_dicts
        
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
        return self._one_type_line_geom_to_instances(centerline_geoms_list)

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

    def get_polyline(self, patch_box, patch_angle, layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        line_list_dic = {}
        records = getattr(self.map_explorer.map_api, layer_name)
        for ind, record in enumerate(records):
            line = self.map_explorer.map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                line = to_patch_coord(line, patch_angle, patch_x, patch_y)
                line_dic = {}
                line_dic['token'] = record['token']
                line_dic['geom'] = line
                line_dic['from'] = layer_name
                line_dic['record'] = record
                line_list_dic[record['token']] = line_dic

                if self.delete:
                    self.delete_record.delete_layer_record(
                        layer_name, record["token"])

        return line_list_dic

    def get_map_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_polyline(
                    patch_box, patch_angle, layer_name)
            elif layer_name in self.polygon_classes:
                geoms = self.get_polygone(
                    patch_box, patch_angle, layer_name)

            map_geom.update(geoms)

        return map_geom

    def _valid_polygon(self, polygon, patch, patch_angle, patch_x, patch_y, record, layer_name, polygon_list_dic):
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                polygon = to_patch_coord(
                    polygon, patch_angle, patch_x, patch_y)
                if polygon.geom_type == 'Polygon':
                    polygon = MultiPolygon([polygon])
                polygon_dic = {}
                polygon_dic['token'] = record['token']
                polygon_dic['geom'] = polygon
                polygon_dic['from'] = layer_name
                polygon_dic['record'] = record
                polygon_list_dic[record['token']] = polygon_dic

                if self.delete:
                    self.delete_record.delete_layer_record(
                        layer_name, record["token"])
                    
        return polygon_list_dic

    def get_polygone(self, patch_box, patch_angle, layer_name):
        if layer_name not in self.map_explorer.map_api.lookup_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_explorer.map_api, layer_name)

        polygon_list_dic = {}
        if layer_name == 'drivable_area':
            for ind, record in enumerate(records):
                polygons = [self.map_explorer.map_api.extract_polygon(
                    polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    polygon_list_dic = self._valid_polygon(polygon, patch, patch_angle, patch_x, patch_y, record, layer_name, polygon_list_dic)
        else:
            for ind, record in enumerate(records):
                polygon = self.map_explorer.map_api.extract_polygon(record['polygon_token'])
                polygon_list_dic = self._valid_polygon(polygon, patch, patch_angle, patch_x, patch_y, record, layer_name, polygon_list_dic)

        return polygon_list_dic

    def get_contour_line_w_record(self, patch_box, patch_angle, layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_explorer.map_api, layer_name)

        polygon_list = []
        record_list = []
        for record in records:
            polygon = self.map_explorer.map_api.extract_polygon(
                record['polygon_token'])

            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                  origin=(patch_x, patch_y))
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)
                    record_list.append(record)

        return polygon_list, record_list

    def line_geoms_to_instances(self, geom_dict):
        line_geom_list = []

        if 'divider' in geom_dict.keys():
            for divider_dic in geom_dict['divider'].values():
                line_geom_list.append(divider_dic['geom'])
            
        if 'lane' in geom_dict.keys():
            for divider_dic in geom_dict['lane'].values():
                for divider_name in ['left_lane_divider', 'right_lane_divider']:
                    if divider_name in divider_dic.keys():
                        line = divider_dic[divider_name]
                        if not line.is_empty:
                            if line.geom_type == 'MultiLineString':
                                for single_line in line.geoms:
                                    # line_instances.append(single_line)
                                    common_result = check_divider_common(single_line, line_geom_list)
                                    if common_result:
                                        line_geom_list = common_result[1]
                                        line_geom_list.append(common_result[0])
                                    else:
                                        line_geom_list.append(single_line)
                            elif line.geom_type == 'LineString':
                                # line_instances.append(line)
                                common_result = check_divider_common(line, line_geom_list)
                                if common_result:
                                    line_geom_list = common_result[1]
                                    line_geom_list.append(common_result[0])
                                else:
                                    line_geom_list.append(line)
                            else:
                                # raise NotImplementedError
                                continue
                        # line_geom = check_divider_common(divider_dic[divider_name], line_geom)
                        # line_geom.append(divider_dic[divider_name])

        new_lans = [lane_dic['geom'] for lane_dic in geom_dict['lane'].values() if lane_dic['from'] in ['new']]
        line_instances = []
        for divider in line_geom_list:
            if not divider.is_empty:
                for lane in new_lans:
                    divider = interpolate(divider)
                    line_instances += keep_non_intersecting_parts(divider, lane, True)
                    break
                    
                line_instances.append(divider)
        
        # line_instances = self._one_type_line_geom_to_instances(line_geom)

        return line_instances

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = []

        for ped_dic in ped_geom.values():
            if ped_dic['geom'].geom_type in ['Polygon', 'MultiPolygon']:
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

        return self._one_type_line_geom_to_instances(results)

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

        return self._one_type_line_geom_to_instances(results)

    def get_org_info_dict(self, map_ins_dict):
        corr_dict = {'divider': [], 'ped_crossing': [], 'boundary': []}
        len_dict = {'divider': 0, 'ped_crossing': 0, 'boundary': 0}

        for vec_class in map_ins_dict.keys():
            if len(map_ins_dict[vec_class]):
                len_dict[vec_class] = len(map_ins_dict[vec_class])
                corr_dict[vec_class] = [
                    i for i in range(len(map_ins_dict[vec_class]))]

        return corr_dict, len_dict

    def get_trans_instance(self, map_ins_dict, trans_args, patch_box, patch_angle):

        corr_dict, len_dict = self.get_org_info_dict(map_ins_dict)

        if trans_args.del_ped[0]:
            map_ins_dict, corr_dict = self.map_trans.delete_layers(
                map_ins_dict, corr_dict, len_dict, 'ped_crossing', trans_args.del_ped)

        if trans_args.shi_ped[0]:
            map_ins_dict, corr_dict = self.map_trans.shift_layers(
                map_ins_dict, corr_dict, len_dict, 'ped_crossing', trans_args.shi_ped,  patch_box)

        if trans_args.add_ped[0]:
            map_ins_dict, corr_dict = self.map_trans.add_layers(
                map_ins_dict, corr_dict, len_dict, 'ped_crossing', trans_args.add_ped,  patch_box, patch_angle)

        if trans_args.del_div[0]:
            map_ins_dict, corr_dict = self.map_trans.delete_layers(
                map_ins_dict, corr_dict, len_dict, 'divider', trans_args.del_div)

        if trans_args.shi_div[0]:
            map_ins_dict, corr_dict = self.map_trans.shift_layers(
                map_ins_dict, corr_dict, len_dict, 'divider', trans_args.shi_div,  patch_box)

        if trans_args.add_div[0]:
            pass  # TODO

        if trans_args.del_bou[0]:
            map_ins_dict, corr_dict = self.map_trans.delete_layers(
                map_ins_dict, corr_dict, len_dict, 'boundary', trans_args.del_bou)

        if trans_args.shi_bou[0]:
            map_ins_dict, corr_dict = self.map_trans.shift_layers(
                map_ins_dict, corr_dict, len_dict, 'boundary', trans_args.shi_bou,  patch_box)

        if trans_args.add_bou[0]:
            map_ins_dict, corr_dict = self.map_trans.add_layers(
                map_ins_dict, corr_dict, len_dict, 'boundary', trans_args.add_bou,  patch_box, patch_angle)

        # if trans_args.wid_bou[0]:
        #     map_ins_dict, corr_dict = self.map_trans.zoom_layers(
        #         map_ins_dict, corr_dict, len_dict, 'boundary', trans_args.wid_bou,  patch_box)

        # if trans_args.aff_tra_pat[0] or trans_args.rot_pat[0] or trans_args.sca_pat[0] or trans_args.ske_pat[0] or trans_args.shi_pat[0]:
        #     map_ins_dict, corr_dict = self.map_trans.transfor_patch(
        #         map_ins_dict, corr_dict, patch_box, trans_args)

        return map_ins_dict, corr_dict, len_dict

# Asynchronous execution utility from https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


def perturb_map_seq(vector_map, trans_args, info, map_version, visual, trans_dic):
    trans_dic = copy.deepcopy(trans_dic)
    trans_ins, corr_dict, len_dict = vector_map.get_trans_instance(
        trans_dic['map_ins_dict'], trans_args, trans_dic['patch_box'], trans_dic['patch_angle'])
    info[map_version+'_correspondence'] = corr_dict

    if trans_args.int_num and trans_args.int_ord == 'before':
        trans_np_dict = geom_to_np(
            trans_ins, inter_args=trans_args.int_num)
    else:
        trans_np_dict = geom_to_np(trans_ins)

    if trans_args.wid_bou[0]:
        trans_np_dict = vector_map.map_trans.zoom_patch_by_layers(
            trans_np_dict, len_dict, 'boundary', trans_args.wid_bou, trans_dic['patch_box'])

    if trans_args.def_pat_tri[0]:
        trans_np_dict = vector_map.map_trans.difromate_map(
            trans_np_dict, trans_args.def_pat_tri, trans_dic['patch_box'])

    if trans_args.def_pat_gau[0]:
        trans_np_dict = vector_map.map_trans.guassian_warping(
            trans_np_dict, trans_args.def_pat_gau, trans_dic['patch_box'])

    if trans_args.noi_pat_gau[0]:
        trans_np_dict = vector_map.map_trans.guassian_noise(
            trans_np_dict, trans_args.noi_pat_gau)

    if (trans_args.int_num and trans_args.int_ord) == 'after' or (not trans_args.int_num and trans_args.int_sav):
        trans_np_dict = np_to_geom(trans_np_dict)
        trans_np_dict = geom_to_np(
            trans_np_dict, trans_args.int_num)
        visual.vis_contours(trans_np_dict, trans_dic['patch_box'], map_version)

    elif trans_args.int_num and not trans_args.int_sav:
        visual.vis_contours(trans_np_dict, trans_dic['patch_box'], map_version)
        trans_np_dict = np_to_geom(trans_np_dict)
        trans_ins_np = geom_to_np(trans_ins)
        trans_np_dict = geom_to_np(
            trans_np_dict, trans_ins_np, int_back=True)

    else:
        # not trans_args.int_num and not trans_args.int_sav
        if visual.switch:
            trans_np_dict_4_vis = np_to_geom(trans_np_dict)
            trans_np_dict_4_vis = geom_to_np(
                trans_np_dict_4_vis, trans_args.int_num)
            visual.vis_contours(trans_np_dict_4_vis,
                                trans_dic['patch_box'], map_version)

    info[map_version] = trans_np_dict

    return info

# @background


def perturb_map(vector_map, trans_args, trans_ins, patch_box):
    # trans_dic = copy.deepcopy(trans_dic)
    # trans_ins, corr_dict, len_dict = vector_map.get_trans_instance(
    #     trans_dic['map_ins_dict'], trans_args, trans_dic['patch_box'], trans_dic['patch_angle'])
    # info[map_version+'_correspondence'] = corr_dict

    if trans_args.int_num and trans_args.int_ord == 'before':
        trans_np_dict = geom_to_np(
            trans_ins, inter_args=trans_args.int_num)
    else:
        trans_np_dict = geom_to_np(trans_ins)

    # if trans_args.wid_bou[0]:
    #     trans_np_dict = vector_map.map_trans.zoom_patch_by_layers(
    #         trans_np_dict, len_dict, 'boundary', trans_args.wid_bou, patch_box)

    if trans_args.def_pat_tri[0]:
        trans_np_dict = vector_map.map_trans.difromate_map(
            trans_np_dict, trans_args.def_pat_tri, patch_box)

    if trans_args.def_pat_gau[0]:
        trans_np_dict = vector_map.map_trans.guassian_warping(
            trans_np_dict, trans_args.def_pat_gau, patch_box)

    if trans_args.noi_pat_gau[0]:
        trans_np_dict = vector_map.map_trans.guassian_noise(
            trans_np_dict, trans_args.noi_pat_gau)

    # if (trans_args.int_num and trans_args.int_ord) == 'after' or (not trans_args.int_num and trans_args.int_sav):
    #     trans_np_dict = np_to_geom(trans_np_dict)
    #     trans_np_dict = geom_to_np(
    #         trans_np_dict, trans_args.int_num)
    #     # visual.vis_contours(trans_np_dict, trans_dic['patch_box'], map_version)

    # elif trans_args.int_num and not trans_args.int_sav:
    #     # visual.vis_contours(trans_np_dict, trans_dic['patch_box'], map_version)
    #     trans_np_dict = np_to_geom(trans_np_dict)
    #     trans_ins_np = geom_to_np(trans_ins)
    #     trans_np_dict = geom_to_np(
    #         trans_np_dict, trans_ins_np, int_back=True)

    return trans_np_dict

# main function


class get_vec_map():
    def __init__(self, info, nusc, nusc_maps, map_explorer, e2g_translation, e2g_rotation, pc_range, out_path, out_type, avm=None, vis=True):
        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.map_explorer = map_explorer
        self.info = info
        self.vis_switch = vis
        self.e2g_translation = e2g_translation
        self.e2g_rotation = e2g_rotation
        self.pc_range = pc_range
        self.avm = avm

        self.vec_classes = ['boundary', 'lane', 'divider',
                            'ped_crossing', 'centerline', 'agent']
        self.info['order'] = ['boundary', 'lane', 'divider',
                              'ped_crossing', 'centerline'] #, 'agent']
        self.get_patch_info()

        vis_path = os.path.join(out_path, 'visualization')
        self.visual = RenderMap(self.info, self.vector_map, vis_path, vis)

        # map info setting
        if out_type == 'json':
            self.map_path = os.path.join(out_path, 'MME_map')
        else:
            self.map_path = None

    def get_patch_info(self):
        patch_h = self.pc_range[4]-self.pc_range[1]
        patch_w = self.pc_range[3]-self.pc_range[0]
        self.patch_size = (patch_h, patch_w)

        map_pose = self.e2g_translation[:2]
        self.patch_box = (map_pose[0], map_pose[1],
                          self.patch_size[0], self.patch_size[1])

        if self.avm is not None:
            rotation = Quaternion._from_matrix(self.e2g_rotation)
            from av2.geometry.se3 import SE3
            city_SE2_ego = SE3(self.e2g_rotation, self.e2g_translation)
            self.ego_SE3_city = city_SE2_ego.inverse()
        else:
            rotation = Quaternion(self.e2g_rotation)
            self.ego_SE3_city = None

        self.patch_angle = quaternion_yaw(rotation) / np.pi * 180
        # map class setting
        self.vector_map = VectorizedLocalMap(
            self.nusc, self.nusc_maps, self.map_explorer, self.patch_box, self.patch_angle, self.avm, self.ego_SE3_city, self.vec_classes)

    def get_map_ann(self, pertube_vers):
        '''get transformed gt map layers'''
        map_geom_org_dic = {}

        # get geom for layers and transfer linestring geom to instance
        if self.info['dataset'] == 'av2':    ## AV2 
            for vec_class in self.vec_classes:
                if vec_class == 'ped_crossing':  # oed_crossing
                    map_geom_org_dic[vec_class] = self.vector_map.extract_local_ped_crossing(
                        self.avm, self.ego_SE3_city, self.patch_box, self.patch_angle)
                elif vec_class == 'boundary':  # road_segment
                    map_geom_org_dic[vec_class] = self.vector_map.extract_local_boundary(
                        self.avm, self.ego_SE3_city, self.patch_box, self.patch_angle)
                elif vec_class == 'lane':  # lane, connector
                    map_geom_org_dic[vec_class] = self.vector_map.extract_local_lane(
                        self.avm, self.patch_box, self.patch_angle)
                elif vec_class == 'centerline':  # lane, connector
                    map_geom_org_dic = self.vector_map.extract_local_centerline(
                        self.avm, self.ego_SE3_city, self.patch_box, self.patch_angle, map_geom_org_dic)
                elif vec_class == 'divider':  # road_divider, lane_divider
                    map_geom_org_dic = self.vector_map.extract_local_divider(
                        self.ego_SE3_city, self.patch_box, self.patch_angle, map_geom_org_dic)
                elif vec_class == 'agent':
                    map_geom_org_dic[vec_class] = {}  # TODO
                else:
                    raise ValueError(f'WRONG vec_class: {vec_class}')
        elif self.info['dataset'] == 'nuscenes': ## NuScenes
            for vec_class in self.vec_classes:
                if vec_class == 'boundary':  # road_segment
                    map_geom_org_dic[vec_class] = self.vector_map.get_map_geom(
                        self.patch_box, self.patch_angle, ['road_segment'])
                elif vec_class == 'lane':  # road_divider, lane_divider
                    map_geom_org_dic[vec_class] = self.vector_map.get_map_geom(
                        self.patch_box, self.patch_angle, ['lane', 'lane_connector'])
                elif vec_class == 'divider':  # road_divider, lane_divider
                    map_geom_org_dic[vec_class] = self.vector_map.get_map_geom(
                        self.patch_box, self.patch_angle, ['road_divider']) #, 'lane_divider'])
                    map_geom_org_dic['lane'] = self.vector_map._get_lane_divider(map_geom_org_dic['lane'])
                elif vec_class == 'ped_crossing':  # oed_crossing
                    map_geom_org_dic[vec_class] = self.vector_map.get_map_geom(
                        self.patch_box, self.patch_angle, ['ped_crossing'])
                elif vec_class == 'centerline':  # lane, connector
                    map_geom_org_dic[vec_class] = self.vector_map._get_centerline(map_geom_org_dic['lane'])
                    map_geom_org_dic = self.vector_map.get_centerline_line(map_geom_org_dic)
                elif vec_class == 'agent':
                    agents_trajectory = get_nuscenes_trajectory(
                        self.nusc, self.info['token'], ['vehicle'], type='geom')
                    map_geom_org_dic[vec_class] = add_tra_to_vecmap(
                        agents_trajectory, self.map_explorer, self.patch_box, self.patch_angle)
                    map_geom_org_dic = get_agent_info(map_geom_org_dic)
                else:
                    raise ValueError(f'WRONG vec_class: {vec_class}')
        elif self.info['dataset'] == 'mme':
            ns_map = NuScenesMap4MME(self.nusc_maps.dataroot, self.nusc_maps.map_name)
            ns_mapex = NuScenesMapExplorer(ns_map)
            self.vector_map = VectorizedLocalMap(
                self.nusc, ns_map, ns_mapex, self.patch_box, self.patch_angle, self.avm, self.ego_SE3_city, self.vec_classes)
            map_geom_org_dic = self.vector_map.gen_vectorized_samples_by_pt_json()
        else:
            sys.exit('wrong dataset')
    
        self.info['map_geom_org_dic'] = map_geom_org_dic
        
        ann_name = 'annotation'
        map_json_name = 'mme'
        for ind, map_v in enumerate(pertube_vers):
            if map_v:
                ann_name = ann_name + '_' + map_v.pt_name
                map_json_name = map_json_name + '_' + map_v.pt_name

            trans_dic = self.vector_map.gen_vectorized_samples(
                map_geom_org_dic, map_v)

            # if self.vis_switch:
            #     trans_np_dict_4_vis = geom_to_np(trans_dic['map_ins_dict'], 20)
            #     self.visual.vis_contours(trans_np_dict_4_vis, self.patch_box, ann_name)
             
            trans_np_dict  = perturb_map(self.vector_map, map_v, trans_dic['map_ins_dict'], self.patch_box)
            
            if map_v.int_num and map_v.int_ord == 'before':
                trans_np_dict = geom_to_np(trans_dic['map_ins_dict'], inter_args=map_v.int_num)
            else:
                trans_np_dict = geom_to_np(trans_dic['map_ins_dict'])

            if map_v.def_pat_tri[0]:
                trans_np_dict = self.vector_map.map_trans.difromate_map(trans_np_dict, map_v.def_pat_tri, self.patch_box)

            if map_v.def_pat_gau[0]:
                trans_np_dict = self.vector_map.map_trans.guassian_warping(trans_np_dict, map_v.def_pat_gau, self.patch_box)

            if map_v.noi_pat_gau[0]:
                trans_np_dict = self.vector_map.map_trans.guassian_noise(trans_np_dict, map_v.noi_pat_gau)
                
            if (map_v.int_num and map_v.int_ord) == 'after' or (not map_v.int_num and map_v.int_sav):
                trans_np_dict = np_to_geom(trans_np_dict)
                trans_np_dict = geom_to_np(
                    trans_np_dict, map_v.int_num)
                if self.vis_switch:
                    self.visual.vis_contours(trans_np_dict, self.patch_box, ann_name)

            elif map_v.int_num and not map_v.int_sav:
                if self.vis_switch:
                    self.visual.vis_contours(trans_np_dict, self.patch_box, ann_name)
                trans_np_dict = np_to_geom(trans_np_dict)
                trans_ins_np = geom_to_np(trans_dic['map_ins_dict'])
                trans_np_dict = geom_to_np(
                    trans_np_dict, trans_ins_np, int_back=True)

            else:  # not map_v.int_num and not map_v.int_sav
                if self.vis_switch:
                    trans_np_dict_4_vis = np_to_geom(trans_np_dict)
                    trans_np_dict_4_vis = geom_to_np(
                        trans_np_dict_4_vis, 20)
                    self.visual.vis_contours(trans_np_dict_4_vis, self.patch_box, ann_name)

            self.info[ann_name] = trans_np_dict
            
            if self.map_path is not None:
                self.info[map_json_name] = vector_to_map_json(
                    trans_dic, self.info, map_json_name, self.map_path)

        # w loop
        # loop = asyncio.get_event_loop()                                              # Have a new event loop
        # looper = asyncio.gather(*[perturb_map(self.vector_map, trans_args, self.info, 'annotation_{}'.format(i), visual, trans_dic) for i in range(10)])         # Run the loop
        # results = loop.run_until_complete(looper)

        # sys.exit('DONE!')

        return self.info

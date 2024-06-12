import copy
import json
import math
import os
import pickle
import random
import secrets
import shutil
import sys
import warnings
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely import ops, affinity
import networkx as nx
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.ops import linemerge, unary_union
from shapely.validation import make_valid


def emtpy_dictory(dic_path):
    shutil.rmtree(dic_path)
    os.mkdir(dic_path)


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def get_length(geom):
    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise ValueError("Input must be a Shapely Polygon or MultiPolygon")

    if isinstance(geom, MultiPolygon):
        geom = ops.unary_union(geom)

    # Get the bounding box coordinates
    min_rect = geom.minimum_rotated_rectangle
    # rect_coords = list(min_rect.exterior.coords)
    rect_coords = list(min_rect.exterior.coords)

    # Calculate length and width
    length_1 = calculate_distance(rect_coords[0], rect_coords[1])
    length_2 = calculate_distance(rect_coords[1], rect_coords[2])

    if length_1 < length_2:
        return [rect_coords[1], rect_coords[2], rect_coords[3], rect_coords[0]]

    return rect_coords[:4]


def token_generator(token_type=[4, 2, 2, 2, 8]):
    """generate a token with a type:
    Ex. '3e8ea889-540c-2113-ae03-f5873a7b6c1ea070'"""

    token = '-'.join([secrets.token_hex(i) for i in token_type])

    return token


class record_generator:
    def __init__(self, pertu_nusc_infos):
        self.pertu_nusc_infos = pertu_nusc_infos

    def node_record_generator(self, coord):
        """{
        "token": "8260ea97-fd4e-4702-b5e8-f1f4465f286f",
        "x": 772.859616346946,
        "y": 1867.5701729191703
        },
        """

        record = {}
        token = token_generator()
        record["token"] = token
        record["x"] = float(coord[0])
        record["y"] = float(coord[1])

        self.pertu_nusc_infos["node"].append(record)

        return record

    def line_record_generator(self, geom):
        """    {
        "token": "97b57a20-185c-4b0f-b405-cc34bf35b1a9",
        "node_tokens": [
        "39878431-98e2-4070-a512-d2c76dcaac4c",
        "79093cc8-e5c4-458d-8d9d-96ce83fdc4af",
        "17a1446a-7dc0-45c5-9354-515455193e5b",
        "85a3cb8b-df2c-456a-a5c8-88e3ec25b4c2",
        "ac669ce0-ced1-4989-bc24-78647f17f30a",
        "5ee9e3a6-f9ea-485c-a916-b6422102467a",
        "12f6f131-97c4-40b4-a85b-4141cc74bf1d",
        "a7278b89-eddb-4f26-89dc-f1a97dec05f8",
        "db854d3d-27e2-470f-9de7-0fa383534558",
        "468e45e8-9793-46b4-a604-6ee8433531dd",
        "9f475eba-75e8-4e9f-b1a1-ac1e1b821ffa",
        "139e0ba6-d96a-4cf9-839d-df4211b291e3",
        "8e91fba4-8573-424c-8c62-baa8acaf7d0f"
          ]
        },
        """

        record = {}

        token = token_generator()
        record["token"] = token

        record["node_tokens"] = []
        for c in geom.coords:
            n_record = self.node_record_generator(c)
            record["node_tokens"].append(n_record["token"])

        self.pertu_nusc_infos["line"].append(record)

        return record

    def polygon_record_generator(self, geom, fake=False):
        """{
        "token": "02eaba43-235f-4b77-99ad-5d44591e315d",
        "exterior_node_tokens": [
            "95375fac-e47e-48e0-9bca-72ccd86efa89",
            "c1b0dd41-8131-4a53-b644-08a9f43343f6",
            "d2f3f652-1179-40fe-923e-61b824d600fe",
            "177dd72d-768e-4fde-a5dd-e950a6c47643"
        ],
        "holes": []
        }
        """
        record = {}

        token = token_generator(token_type=[4, 2, 2, 2, 8])
        record["token"] = token

        record["exterior_node_tokens"] = []
        record["holes"] = []

        if not fake:
            exterior_nodes = geom.exterior.coords
            for n in exterior_nodes:
                n_record = self.node_record_generator(n)
                record["exterior_node_tokens"].append(n_record["token"])

            holes = geom.interiors
            for h in holes:
                hole = {"node_tokens": []}
                for n in h.coords:
                    n_record = self.node_record_generator(n)
                    hole["node_tokens"].append(n_record["token"])

                record["holes"].append(hole)

        self.pertu_nusc_infos["polygon"].append(record)

        return record

    def layer_record_generator(self, layer_name, geom_dic, fake=False):
        record = {}

        if layer_name == 'boundary':
            geom = geom_dic
            token = token_generator()
            record["token"] = token
        else:
            geom = geom_dic['geom']
            record['token'] = geom_dic['token']            

        if geom.geom_type == 'LineString':
            line_record = self.line_record_generator(geom)
            record["line_token"] = line_record["token"]
        elif geom.geom_type == 'Polygon':
            polygon_record = self.polygon_record_generator(geom)
            record["polygon_token"] = polygon_record["token"]
        else:
            warnings.warn(
                "Warning...........geom type is neither LineString nor Polygon")
            return None
        
        if layer_name not in ['centerline', 'agent', 'boundary']:
            record['from'] = geom_dic['from']            
        
        if layer_name in ['lane', 'ped_crossing']:
            try:
                record['centerline_token'] = geom_dic['centerline_token']
            except:
                record['centerline_token'] = None
        
        if layer_name == 'centerline':
            for k in ['lane_token', 'ped_crossing_token']:
                if k in geom_dic.keys():
                    record[k] = geom_dic[k]            
                
        if fake:
            if layer_name in ["ped_crossing", 'boundary']:
                polygon_record = self.polygon_record_generator(geom, fake)
                record["polygon_token"] = polygon_record["token"]

        self.pertu_nusc_infos[layer_name].append(record)

        return record


def vector_to_map_json(info_dic, info, map_version, save=None, fake=True):
    r_gen = record_generator(info_dic["pertu_nusc_infos"])

    ins_dic = info_dic["map_ins_dic_patch"]
    geom_dic = info_dic['map_geom_dic_patch']
    for layer_name in geom_dic.keys():
        if layer_name in ['boundary']:
            for geom in ins_dic[layer_name]:
                r_gen.layer_record_generator(layer_name, geom)
        else:
            for geom in geom_dic[layer_name].values():
                r_gen.layer_record_generator(layer_name, geom)

    if save is not None:
        save_path = os.path.join(
            save, info['scene_token'], info['token'], map_version, "maps", "expansion")
        check_path(save_path)
        with open(os.path.join(save_path, "singapore-onenorth.json"), "w") as outfile:
            json.dump(r_gen.pertu_nusc_infos, outfile)

    return r_gen.pertu_nusc_infos

def delete_elements_by_indices(data_list, indices):
    # Ensure indices are sorted in descending order to avoid shifting issues
    indices_sorted = sorted(indices, reverse=True)
    
    # Delete elements at the specified indices
    for index in indices_sorted:
        if 0 <= index < len(data_list):
            del data_list[index]
    
    return data_list


class delet_record:
    def __init__(self, map_explorerm,
                 pertu_nusc_infos) -> None:
        self.map_explorerm = map_explorerm
        self.pertu_nusc_infos = pertu_nusc_infos

    def delete_node_record(self, token) -> None:
        n_ind = self.map_explorerm.map_api.getind("node", token)
        self.pertu_nusc_infos["node"][n_ind] = None

    def delete_line_record(self, token) -> None:
        polygon_record = self.map_explorerm.map_api.get('line', token)

        for n in polygon_record["node_tokens"]:
            self.delete_node_record(n)

        l_ind = self.map_explorerm.map_api.getind("line", token)
        self.pertu_nusc_infos["line"][l_ind] = None

    def delete_polygon_record(self, token) -> None:
        polygon_record = self.map_explorerm.map_api.get('polygon', token)

        for n in polygon_record["exterior_node_tokens"]:
            self.delete_node_record(n)

        for l in polygon_record["holes"]:
            self.delete_node_record(l)

        p_ind = self.map_explorerm.map_api.getind("polygon", token)
        self.pertu_nusc_infos["polygon"][p_ind] = None

    def delete_layer_record(self, layer_name, token) -> None:
        layer_record = self.map_explorerm.map_api.get(layer_name, token)

        if layer_name in ["road_segment", "lane", "ped_crossing"]:
            self.delete_polygon_record(layer_record["polygon_token"])
        elif layer_name in ["road_divider", "lane_divider"]:
            self.delete_line_record(layer_record["line_token"])
        else:
            warnings.warn("Warning...........layer is not ture")

        layer_ind = self.map_explorerm.map_api.getind(layer_name, token)
        self.pertu_nusc_infos[layer_name][layer_ind] = None


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def patch_coords_2_box(coords):
    box = ((coords[0] + coords[2])/2, (coords[1] + coords[3]
                                       )/2, coords[3] - coords[1], coords[2] - coords[0])
    return box


def patch_box_2_coords(box):
    coors = (box[0] - box[3]/2, box[1] - box[2]/2,
             box[0] + box[3]/2, box[1] + box[2]/2)

    return coors


def union_centerline(line_geoms_dic):
    pts_G = nx.DiGraph()
    junction_pts_list = []

    for value in line_geoms_dic.values():
        line_geom = value['centerline']
        if line_geom.geom_type == 'MultiLineString':
            start_pt = np.array(line_geom.geoms[0].coords).round(3)[0]
            end_pt = np.array(line_geom.geoms[-1].coords).round(3)[-1]
            for single_geom in line_geom.geoms:
                single_geom_pts = np.array(single_geom.coords).round(3)
                for idx, pt in enumerate(single_geom_pts[:-1]):
                    pts_G.add_edge(tuple(single_geom_pts[idx]), tuple(
                        single_geom_pts[idx+1]))
        elif line_geom.geom_type == 'LineString':
            centerline_pts = np.array(line_geom.coords).round(3)
            start_pt = centerline_pts[0]
            end_pt = centerline_pts[-1]
            for idx, pts in enumerate(centerline_pts[:-1]):
                pts_G.add_edge(tuple(centerline_pts[idx]), tuple(
                    centerline_pts[idx+1]))
        else:
            raise NotImplementedError
        valid_incoming_num = 0
        for idx, pred in enumerate(value['incoming_tokens']):
            if pred in line_geoms_dic.keys():
                valid_incoming_num += 1
                pred_geom = line_geoms_dic[pred]['centerline']
                if pred_geom.geom_type == 'MultiLineString':
                    pred_pt = np.array(pred_geom.geoms[-1].coords).round(3)[-1]
                    pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
                else:
                    pred_pt = np.array(pred_geom.coords).round(3)[-1]
                    pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
        if valid_incoming_num > 1:
            junction_pts_list.append(tuple(start_pt))

        valid_outgoing_num = 0
        for idx, succ in enumerate(value['outgoing_tokens']):
            if succ in line_geoms_dic.keys():
                valid_outgoing_num += 1
                succ_geom = line_geoms_dic[succ]['centerline']
                if succ_geom.geom_type == 'MultiLineString':
                    succ_pt = np.array(succ_geom.geoms[0].coords).round(3)[0]
                    pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
                else:
                    succ_pt = np.array(succ_geom.coords).round(3)[0]
                    pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
        if valid_outgoing_num > 1:
            junction_pts_list.append(tuple(end_pt))

    roots = (v for v, d in pts_G.in_degree() if d == 0)
    leaves = [v for v, d in pts_G.out_degree() if d == 0]
    all_paths = []
    for root in roots:
        for leave in leaves:
            paths = nx.all_simple_paths(pts_G, root, leave)
            all_paths.extend(paths)

    final_line_paths = []
    for path in all_paths:
        merged_line = LineString(path)
        merged_line = merged_line.simplify(0.2, preserve_topology=True)
        final_line_paths.append(merged_line)
    return final_line_paths


def union_line(line_geoms):
    pts_G = nx.DiGraph()
    # junction_pts_list = []
    for line_geom in line_geoms:
        if line_geom.geom_type == 'MultiLineString':
            # start_pt = np.array(line_geom.geoms[0].coords).round(3)[0]
            # end_pt = np.array(line_geom.geoms[-1].coords).round(3)[-1]
            for single_geom in line_geom.geoms:
                single_geom_pts = np.array(single_geom.coords).round(3)
                for idx, pt in enumerate(single_geom_pts[:-1]):
                    pts_G.add_edge(tuple(single_geom_pts[idx]), tuple(
                        single_geom_pts[idx+1]))
        elif line_geom.geom_type == 'LineString':
            centerline_pts = np.array(line_geom.coords).round(3)
            # start_pt = centerline_pts[0]
            # end_pt = centerline_pts[-1]
            for idx, pts in enumerate(centerline_pts[:-1]):
                pts_G.add_edge(tuple(centerline_pts[idx]), tuple(
                    centerline_pts[idx+1]))
        else:
            raise NotImplementedError

    roots = (v for v, d in pts_G.in_degree() if d == 0)
    leaves = [v for v, d in pts_G.out_degree() if d == 0]
    all_paths = []
    for root in roots:
        for leave in leaves:
            paths = nx.all_simple_paths(pts_G, root, leave)
            all_paths.extend(paths)

    final_line_paths = []
    for path in all_paths:
        merged_line = LineString(path)
        merged_line = merged_line.simplify(0.2, preserve_topology=True)
        final_line_paths.append(merged_line)
    return final_line_paths


def keep_non_intersecting_parts(geom_keep, geom_intersect, edge_include=False):
    # Check if the line intersects with the polygon
    if not geom_keep.intersects(geom_intersect):
        return [geom_keep]

    # If an intersection occurs, find the difference
    intersecting_part = geom_keep.intersection(geom_intersect)
    difference = geom_keep.difference(intersecting_part)

    if edge_include:
        interecting_edge = None
        if geom_intersect.geom_type == 'MultiPolygon':
            geom_exteriors = unary_union([geom.exterior for geom in geom_intersect.geoms])
            interecting_edge = geom_keep.intersection(geom_exteriors)
        elif geom_intersect.geom_type == 'Polygon':
            interecting_edge = geom_keep.intersection(geom_intersect.exterior)
            
        if interecting_edge:
            if interecting_edge.geom_type == 'LineString':
                difference = linemerge(difference, interecting_edge)

    # If the difference is a MultiLineString, return its components
    if difference.geom_type in ['MultiLineString', 'MultiPolygon']:
        return list(difference.geoms)
    elif difference.geom_type in ['LineString', 'Polygon']:
        return [difference]
    else:
        return []


def to_patch_coord(new_polygon, patch_angle, patch_x, patch_y):

    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                  origin=(patch_x, patch_y), use_radians=False)
    new_polygon = affinity.affine_transform(new_polygon,
                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
    return new_polygon


def get_geom_in_patch(map_explorer, geoms, patch_box=[0, 0, 60, 30], patch_angle=0):
    patch = map_explorer.get_patch_coord(patch_box, patch_angle)

    if isinstance(geoms, dict):
        for geom in geoms.values():
            try:
                geom['geom'] = geom['geom'].intersection(patch)
            except:
                geom['geom'] = make_valid(geom['geom']).intersection(patch)
    elif isinstance(geoms, list):
        new_list = []
        for geom in geoms:
            new_geom = geom.intersection(patch)
            if new_geom:
                if new_geom.geom_type in ['MultiPolygon', 'MultiLineString']:
                    new_list += [gm for gm in new_geom.geoms]
                elif new_geom.geom_type in ['Polygon', 'LineString']:
                    new_list.append(new_geom)
                else:
                    continue
        geoms = new_list
        # for ind in range(len(geoms)):
        #     geoms[ind] = geoms[ind].intersection(patch)
    else:
        sys.exit('wrong data type')

    return geoms

def is_empty(data):
    if isinstance(data, np.ndarray):
        return data.size == 0
    elif isinstance(data, tuple):
        return len(data) == 0
    else:
        raise TypeError("Unsupported type. Only NumPy arrays and tuples are supported.")

def get_index(ins_c, threshold=[None, None]):
    indices_list = []
    if threshold[0]:
        indices = np.where(ins_c > threshold[0])
        try:
            indices_list += list(indices[0]) # Convert the result to a list of index tuples
        except:
            pass
            
    if threshold[1]:
        indices = np.where(ins_c < threshold[1])
        try:
            indices_list += list(indices[0]) # Convert the result to a list of index tuples
        except:
            pass
    indices_list = list(set(indices_list))
    
    return indices_list

def get_vect_in_patch(vect_list, patch_box=[0, 0, 60, 30]):
    x_min = patch_box[0] - patch_box[3] / 2
    x_max = patch_box[0] + patch_box[3] / 2
    y_min = patch_box[1] - patch_box[2] / 2
    y_max = patch_box[1] + patch_box[2] / 2
    xy_range = [[x_min, x_max], [y_min, y_max]]

    new_vect_list = []
    for vect in vect_list:
        
        # for dem in range(vect.shape[1]):
        #     ins_c = vect[:, dem]
        #     indices_list = get_index(ins_c, [xy_range[dem][1], xy_range[dem][0]])
        
        # for indice in indices_list:
        #     np.delete(vect, indice)        
        
        new_vect_list.append(threshold_ins(vect, xy_range))
    
    return new_vect_list

def interpolate(instance, inter_args=0):
    if not inter_args:
        inter_args = math.floor(instance.length)
        if not inter_args:
            return instance
    
    distance = np.linspace(0, instance.length, inter_args)
    instance_coord = [np.array(instance.interpolate(n).coords)[
        0] for n in distance]

    try:
        instance = LineString(instance_coord)
    except:
        pass

    return instance


def update_lane(lane_and_connectors, incoming_left, outgoing_left, new_lane_and_connectors, new_lanes, ck):
    lane_and_connectors[ck]['incoming_tokens'] = incoming_left
    lane_and_connectors[ck]['outgoing_tokens'] = outgoing_left
    new_lane_and_connectors[ck] = lane_and_connectors[ck]
    new_lane_token = incoming_left + outgoing_left
    for lane in new_lane_token:
        new_lane_and_connectors[lane] = lane_and_connectors[lane]
        new_lanes.append(lane_and_connectors[lane]['polygon'])

    return new_lane_and_connectors, new_lanes

def get_intersection(geom, it_geom, type='multi', keep_it_only=False):
    new_ccgd = []
    for ccgd_g in geom.geoms:
        for c_g in it_geom:
            if ccgd_g.intersection(c_g):
                if keep_it_only:
                    new_ccgd.append(ccgd_g.intersection(c_g))
                else:
                    new_ccgd.append(ccgd_g)
                    
    return new_ccgd

def check_isolated_new(check_geom_dict, intersect_geom_dict_list=[], keep_intersect_only=False):
    if bool(check_geom_dict):
        
        copied_cgd = copy.deepcopy(check_geom_dict)
        
        # conbine checker
        checker_geom = []
        for check_layer in intersect_geom_dict_list:
            if len(check_layer):
                checker_geom.append(unary_union([check_copied_cgd['geom'] for check_copied_cgd in check_layer.values()]))
        
        # if not len(checker_geom):
        #     return {}
            
        # checker_geom = unary_union(checker_geom)
        
        for ccgd in copied_cgd.values():
            if ccgd['geom'].geom_type == 'MultiLineString':
                new_ccgd = get_intersection(ccgd['geom'], checker_geom, keep_it_only=keep_intersect_only)
                if len(new_ccgd) > 1:
                   check_geom_dict[ccgd['token']]['geom'] = unary_union(new_ccgd)
                elif len(new_ccgd) == 1:
                   check_geom_dict[ccgd['token']]['geom'] = new_ccgd[0]
                else:
                   check_geom_dict.pop(ccgd['token'])                  
            
            elif ccgd['geom'].geom_type == 'MultiPolygon':
                new_ccgd = get_intersection(ccgd['geom'], checker_geom, keep_it_only=keep_intersect_only)
                if len(new_ccgd) > 1:
                   check_geom_dict[ccgd['token']]['geom'] = unary_union(new_ccgd)
                elif len(new_ccgd) == 1:
                   check_geom_dict[ccgd['token']]['geom'] = new_ccgd[0]
                else:
                   check_geom_dict.pop(ccgd['token'])                  
            
            elif ccgd['geom'].geom_type in ['Polygon', 'LineString']:
                check_it = 0
                for c_g in checker_geom:
                    if ccgd['geom'].intersection(c_g):
                        check_it = 1
                        if keep_intersect_only:
                            check_geom_dict[ccgd['token']]['geom'] = ccgd['geom'].intersection(c_g)
                        break
                
                if not check_it:    
                    check_geom_dict.pop(ccgd['token'])                  

    return check_geom_dict

def get_interect_info(line_dict, geoms_dict, layer_name, geom_new_key, line_new_key=None):
    if line_new_key is None:
        line_new_key = layer_name+'_token'
    else:
        line_new_key = line_new_key + '_' + layer_name + '_token'

    if line_new_key not in line_dict:
        line_dict[line_new_key] = []
    
    for layer_dic in geoms_dict[layer_name].values():
        if line_dict['geom'].intersects(layer_dic['geom']):
            line_dict[line_new_key].append(layer_dic['token'])
            if geom_new_key not in layer_dic:
                layer_dic[geom_new_key] = []
            layer_dic[geom_new_key].append(line_dict['token'])

    return line_dict, geoms_dict

def get_path(ls_dict):
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

    local_centerline_paths = []
    for path in all_paths:
        merged_line = LineString(path)
        merged_line = merged_line.simplify(0.2, preserve_topology=True)
        # local_centerline_paths['token'] = token_generator()
        # local_centerline_paths['geom'] = merged_line
        local_centerline_paths.append(merged_line)

    return local_centerline_paths


def proc_polygon(polygon, ego_SE3_city):
    interiors = []
    exterior_cityframe = np.array(list(polygon.exterior.coords))
    exterior_egoframe = ego_SE3_city.transform_point_cloud(exterior_cityframe)
    for inter in polygon.interiors:
        inter_cityframe = np.array(list(inter.coords))
        inter_egoframe = ego_SE3_city.transform_point_cloud(inter_cityframe)
        interiors.append(inter_egoframe[:, :3])

    new_polygon = Polygon(exterior_egoframe[:, :3], interiors)
    return new_polygon


def proc_line(line, ego_SE3_city):
    new_line_pts_cityframe = np.array(list(line.coords))
    new_line_pts_egoframe = ego_SE3_city.transform_point_cloud(
        new_line_pts_cityframe)
    line = LineString(new_line_pts_egoframe[:, :3])  # TODO
    return line


def valid_geom(geom, map_explorer, patch_box, patch_angle):
    patch = map_explorer.get_patch_coord(patch_box, patch_angle)

    if geom.is_valid:
        new_geom = geom.intersection(patch)
        if not new_geom.is_empty:
            # region
            # if new_geom.geom_type == 'MultiLineString':
            #     inter_points = geom.intersection(patch.boundary)
            #     ip_list = []
            #     connect_lines = []
            #     for p in inter_points.geoms:
            #         ip_list.append(p)
            #     connect_lines.append(LineString(ip_list))
            #     for l in new_geom.geoms:
            #         connect_lines.append(l)

            #     new_multi_lines = MultiLineString(connect_lines)
            #     new_geom = ops.linemerge(new_multi_lines)

            # if new_geom.geom_type == 'MultiLineString':
            #     correct_coords = self.fix_corner(np.array(geom.coords), patch_box)
            #     new_geom = LineString([[x[0], x[1]] for x in correct_coords])

            # if new_geom.geom_type == 'MultiLineString':
            #     return None
            # endregion

            new_geom = affinity.rotate(
                new_geom, -patch_angle, origin=(patch_box[0], patch_box[1]))
            new_geom = affinity.affine_transform(new_geom,
                                                 [1.0, 0.0, 0.0, 1.0, -patch_box[0], -patch_box[1]])

        else:
            return None
    else:
        # print('geom is not valid.')
        return None

    return new_geom


def get_centerline_info(centerline_dict, geoms_dict):

    centerline_dict, geoms_dict = get_interect_info(centerline_dict, geoms_dict, 'lane', 'centerline_token')
    centerline_dict, geoms_dict = get_interect_info(centerline_dict, geoms_dict, 'ped_crossing', 'centerline_token')

    return centerline_dict, geoms_dict


def move_polygon_away(polygon_to_move, other_polygons, direction, step_size=0.1):
    """
    Move a Shapely Polygon away from other polygons in the specified direction.

    Parameters:
    - polygon_to_move (Polygon): The Shapely Polygon to be moved.
    - other_polygons (list): A list of other Shapely Polygons that may cover the polygon_to_move.
    - direction (tuple): A tuple representing the direction vector in which to move the polygon (e.g., (1, 0) for right).
    - step_size (float): The step size for each move.

    Returns:
    - moved_polygon (Polygon): The moved Shapely Polygon.
    """

    moved_distance = 0
    # Iterate until the polygon is no longer covered
    while polygon_to_move.intersects(other_polygons):
        # Move the polygon in the specified direction
        polygon_to_move = affinity.translate(
            polygon_to_move, direction[0] * step_size, direction[1] * step_size)
        moved_distance += step_size

    return polygon_to_move, moved_distance


def fix_corner(vect, patch_box):
    x_min = patch_box[0] - patch_box[3] / 2
    x_max = patch_box[0] + patch_box[3] / 2
    y_min = patch_box[1] - patch_box[2] / 2
    y_max = patch_box[1] + patch_box[2] / 2
    xy_range = [[x_min, x_max], [y_min, y_max]]

    return threshold_ins(vect, xy_range)

def threshold_ins(vect, xy_range):
    for dem in range(2):
        ins_c = vect[:, dem]
        indices_list = get_index(ins_c, [xy_range[dem][1], xy_range[dem][0]])
    
    for indice in indices_list:
        np.delete(vect, indice)
        
    return vect        

def get_agent_info(geoms_dict):
    for layer_dic in geoms_dict['agent'].values():
        layer_dic, geoms_dict = get_interect_info(layer_dic, geoms_dict, 'lane', 'agent_token')

        layer_dic, geoms_dict = get_interect_info(layer_dic, geoms_dict, 'lane', 'agent_eco_token', 'eco')

    return geoms_dict


def np_to_geom(map_ins_dict):
    layers = [k for k in map_ins_dict.keys()]
    map_dict = make_dict(layers)
    for vec_class in layers:
        if len(map_ins_dict[vec_class]):
            for instance in map_ins_dict[vec_class]:
                try:
                    instance = LineString([[x[0], x[1]] for x in instance])
                except:
                    continue
                map_dict[vec_class].append(instance)

    return map_dict

def multi_2_single(geom):
    if geom.type == 'MultiLineString':
        geom = ops.linemerge(geom)
    elif geom.type == 'MultiPolygon':
        geom = ops.unary_union(geom)
        
    return geom

def interpolate_geom(geom, inter, inter_acc):
    if geom.geom_type == 'Polygon':
        instance = geom.exterior
    elif geom.geom_type == 'LineString':
        instance = geom
    else:
        return None
    
    if geom.is_empty:
        return None
    
    if inter:
        instance = interpolate(instance, inter_acc)
    
    return instance        

def geom_to_np(map_ins_dict, inter = False, inter_args=0, int_back=False, info=None, map_version=None, save=None):
    """_summary_

    Args:
        map_ins_dict (dict): _description_
        inter_args (int/dict, optional): Interpolation accuracy. If int_back=true, it's a dict with original vector accuracy. Default is 0.
        int_back (bool, optional): Interpolate geometry from vector back to original accuracy. Default is False.
        info (dict, optional): _description_. Defaults to None.
        map_version (_type_, optional): _description_. Defaults to None.
        save (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    map_dict = {}
    for vec_class in map_ins_dict.keys():
        map_dict[vec_class] = []
        
        if len(map_ins_dict[vec_class]):
            for ind, instance in enumerate(map_ins_dict[vec_class]):
                if not instance.is_empty:
                    # aget the interpolation accuracy 
                    if int_back:
                        shape = inter_args[vec_class][ind].shape
                        inter_acc = shape[0]
                    else:
                        inter_acc = inter_args
                    
                    # Managing the multi-geometries
                    if instance.geom_type in ['MultiPolygon', 'MultiLineString']:
                        instance = multi_2_single(instance)
                        
                        if instance.geom_type in ['MultiPolygon', 'MultiLineString']:
                            inter_gs = []
                            for g in instance.geoms:
                                g = interpolate_geom(g, inter, inter_acc)
                                if g:
                                    inter_gs.append(np.array(g.coords))
                            if inter_gs:
                                instance = np.concatenate(inter_gs)
                            else:
                                continue
                        else:
                            instance = interpolate_geom(instance, inter, inter_acc)
                            instance = np.array(instance.coords)
                    
                    elif instance.geom_type in ['Polygon', 'LineString']:
                        instance = interpolate_geom(instance, inter, inter_acc)
                        instance = np.array(instance.coords)
                    
                    else:
                        instance = np.array(instance.coords)
                    
                    if isinstance(instance, np.ndarray) and instance.size != 0:    
                        map_dict[vec_class].append(instance)
                    else:
                        continue

    if save is not None:
        save_path = os.path.join(
            save, info['scene_token'], info['token'], map_version, "maps", "expansion")
        check_path(save_path)
        with open(os.path.join(save_path, "pt_patch.pkl"), "wb") as outfile:
            # json.dump(map_dict, outfile)
            pickle.dump(map_dict, outfile)
            # np.save(outfile, map_dict)

    return map_dict

def affine_transfer_4_add_centerline(new_lane, xoff=0, yoff=0, angle=0, origin_rot='center', xfact=0, yfact=0, origin_sca='center'):
    new_lane = affinity.translate(new_lane, xoff, yoff) #shift
    new_lane = affinity.rotate(new_lane, angle, origin_rot) #rotate
    new_lane = affinity.scale(new_lane, xfact, yfact, origin=origin_sca) #flip
    
    return new_lane

def move_polygons(line, polygons_dic, distance, except_token = []):
    for key, polygon in polygons_dic.items():
        if key in except_token:
            continue
        
        # Creating a polygon object
        poly = polygon['geom']

        # Calculate the intersection of a polygon and a line segment
        intersection = poly.intersection(line)
        # Calculate the unit vector of a line segment
        dx = line.coords[1][0] - line.coords[0][0]
        dy = line.coords[1][1] - line.coords[0][1]
        length = (dx ** 2 + dy ** 2) ** 0.5
        ux = dx / length
        uy = dy / length

        # Calculate the distance moved outward
        move_x = -uy * distance
        move_y = ux * distance
        # If the polygon intersects the line segment
        if intersection:
            if poly.geom_type == 'LineString':
                continue
            # Move polygon vertices outward
            if poly.geom_type == 'MultiPolygon':
                new_poly = []
                for py in poly.geoms:
                    new_coords = [(x + move_x, y + move_y) for x, y in py.exterior.coords]
                    new_poly.append(Polygon(new_coords))
                new_poly = ops.unary_union(new_poly)
            else:
                new_coords = [(x + move_x, y + move_y) for x, y in poly.exterior.coords]
                new_poly = Polygon(new_coords)
        else:
            # If there is no intersection, move directly outward
            new_poly = affinity.translate(poly, xoff=move_x, yoff=move_y)
            
        polygon['geom'] = new_poly

    return polygons_dic

def random_select_element(all_elements, num_elements):
    if isinstance(all_elements, dict):
        keys = list(all_elements.keys())
        selected_keys = random.sample(keys, num_elements)
        return [all_elements[key] for key in selected_keys]
    elif isinstance(all_elements, list):
        return random.sample(all_elements, num_elements)


def randomly_pop_elements(my_list, num_elements):
    popped_elements = []
    for _ in range(num_elements):
        if my_list:  # Ensure the list is not empty
            random_index = random.randint(0, len(my_list) - 1)
            popped_elements.append(my_list.pop(random_index))
    return popped_elements, my_list

class NuScenesMap4MME(NuScenesMap):
    def __init__(self,
                 dataroot: str = '/data/sets/nuscenes',
                 map_name: str = 'singapore-onenorth'):
        
        # super(NuScenesMap4MME, self).__init__(dataroot, map_name)
        super().__init__(dataroot, map_name)
        self.non_geometric_line_layers = ['road_divider', 'lane_divider', 'traffic_light', 'centerline', 'boundary', 'agent', 'divider']
        
        self.boundary = self._load_layer('boundary')
        self.divider = self._load_layer('divider')
        self.centerline = self._load_layer('centerline')
        
        # super(NuScenesMap4Mod, self).__init__(dataroot, map_name)
        
    def _make_shortcuts(self) -> None:
        """ Makes the record shortcuts. """

        # Makes a shortcut between nongeometric records and their nodes.
        for layer_name in self.non_geometric_polygon_layers:
            if layer_name == 'drivable_area':  # Drivable area has more than one geometric representation.
                pass
            else:
                for record in self.__dict__[layer_name]:
                    polygon_obj = self.get('polygon', record['polygon_token'])
                    record['exterior_node_tokens'] = polygon_obj['exterior_node_tokens']
                    record['holes'] = polygon_obj['holes']

        for layer_name in self.non_geometric_line_layers:
            for record in self.__dict__[layer_name]:
                record['node_tokens'] = self.get('line', record['line_token'])['node_tokens']

        # Makes a shortcut between stop lines to their cues; there are different cues for different types of stop lines.
        # Refer to `_get_stop_line_cue()` for details.
        for record in self.stop_line:
            cue = self._get_stop_line_cue(record)
            record['cue'] = cue

        # # Makes a shortcut between lanes to their lane divider segment nodes.
        # for record in self.lane:
        #     record['left_lane_divider_segment_nodes'] = [self.get('node', segment['node_token']) for segment in
        #                                                  record['left_lane_divider_segments']]
        #     record['right_lane_divider_segment_nodes'] = [self.get('node', segment['node_token']) for segment in
        #                                                   record['right_lane_divider_segments']]

def make_dict(my_list = [], val = []):
    # Creating a dictionary with empty lists as values
    my_dict = {}
    for k in my_list:
        my_dict[k] = copy.deepcopy(val)

    # Printing the resulting dictionary
    return my_dict

def layer_dict_generator(geom, token = '', source = ''):
    dic = {}
    dic['geom'] = geom
    dic['from'] = source
    
    if token:
        dic['token'] = token
    else:
        dic['token'] = token_generator()        

    return dic

def check_common(segment_4_lane, lane_divider):
    lane_segments = list(set([seg['node_token'] for seg in segment_4_lane]))
    for divider in lane_divider:
        if len(divider['lane_divider_segments']):
            divider_segments = list(set([seg['node_token'] for seg in divider['lane_divider_segments']]))
            if lane_segments == divider_segments:
                return divider['token']
    
    return None

def count_layer_element(geom_dict):
    num_layer_elements = {}
    for k, v in geom_dict.items():
        num_layer_elements[k] = len(v)
        
    return num_layer_elements


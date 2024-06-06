import math
import random
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from shapely import affinity, ops
from shapely.geometry import Point

from ..utils import *


class PerturbParameters():
    """class for pertubation parameter setting"""
    def __init__(self,
                 pt_name=None, # perturbation version name
                 
                 ## [switch:bool, proportion:flot, parameter]
                 # centerline(lane+divider)
                 del_lan=[0, 0, None], #[switch, proportion, None]
                 add_lan=[0, 0, None], #[switch, proportion, None]
                 wid_lan=[0, 0, 0], #[switch, proportion, distance], the widening distance should be within a reasonable range (-2,2) 
                 rot_lan=[0, 0, [[0,0],[0,0]]], #[switch, proportion, [angle_range, origin]], the origin should be the 'center'.
                 sca_lan=[0, 0, [1,1,'center']], #[switch, proportion, [xfact, yfact, origin]], xfact and yfact should be 1 or -1 to perform a flip.
                 shi_lan=[0, 0, [[0,0],[0,0]]], #[switch, proportion, [xoff_range, yoff_range]]
                 
                 # ped_crossing perturbation
                 del_ped=[0, 0, None],  #[switch, proportion, None]
                 add_ped=[0, 0, None],  #[switch, proportion, None] 
                 shi_ped=[0, 0, 0], #[switch, proportion, distance]
                 
                 # dividers perturbation
                 del_div=[0, 0, None],  #[switch, proportion, None]
                 
                 # boundray perturabtion
                 wid_bou=[0, 0, 0], #[switch, proportion, distance], the widening distance should be within a reasonable range (-2,15)
                 
                 # patch perturbation
                 rot_pat=[0, None, [[0,0], [0, 0]]],  #[switch, proportion, [angle_range, origin]], the origin should be the center coordinates of the patch_box.
                 sca_pat=[0, None, [1, 1, (0, 0)]],  #[switch, proportion, [xfact, yfact, origin]], xfact and yfact should be 1 or -1 to perform a flip.
                 shi_pat=[0, None, [[0,0], [0,0]]],  #[switch, proportion, [xoff_range, yoff_range]]
                 
                 def_pat_tri=[0, None, [0, 0, 0]], #[switch, proportion, [xamp, yamp, ramp]], Horizontal, Vertical, and Inclination distortion amplitude
                 def_pat_gau=[0, None, [0, 1]], # [switch, proportion, [mean, standard]]
                 noi_pat_gau=[0, None, [0, 1]], # [switch, proportion, [mean, standard]]
                 
                 ## other pertubation setting
                 diy=False, #Manually select the layers that need to be perturbated
                 truncate=False, #Visualizing the intermediate steps of perturbations
                 # Interpolation
                 int_num=0, #Interpolation density
                 int_ord='before', #Before the perturbation or after it
                 int_sav=False):  #Save the interpolated instances
        
        self.pt_name = pt_name
        
        self.del_ped = del_ped
        self.shi_ped = shi_ped
        self.add_ped = add_ped
        self.del_div = del_div
        self.del_lan = del_lan
        self.wid_lan = wid_lan
        self.add_lan = add_lan
        self.rot_lan = rot_lan
        self.sca_lan = sca_lan
        self.shi_lan = shi_lan
        self.wid_bou = wid_bou
        self.rot_pat = rot_pat
        self.sca_pat = sca_pat
        self.shi_pat = shi_pat
        self.def_pat_tri = def_pat_tri
        self.def_pat_gau = def_pat_gau
        self.noi_pat_gau = noi_pat_gau

        self.diy = diy
        self.truncate = truncate
        self.int_num = int_num
        self.int_ord = int_ord
        self.int_sav = int_sav

    def update_attribute(self, attr_name, new_value):
        """Modify variables 

        Args:
            attr_name (str): variable name
            new_value (_type_): varibale value

        Raises:
            AttributeError: if there is no variable
        """
        if hasattr(self, attr_name):
            setattr(self, attr_name, new_value)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr_name}'")     
class MapTransform:
    """map pertubation class"""
    def __init__(self, map_explorer: NuScenesMapExplorer, geom_dict, vector_map=None, visual=None, tran_args=None, patch_angle=0, patch_box=[0,0,60,30]):
        """initialization

        Args:
            map_explorer (NuScenesMapExplorer): Dataset class in the nuScenes map dataset explorer.
            geom_dict (dict): geomtry map layers
            vector_map (VectorizedMap, optional): class of vectorize geometry map. Defaults to None.
            visual (RenderMap, optional): class in map visualization. Defaults to None.
            tran_args (dict, optional): pertubation parameters. Defaults to None.
            patch_angle (int, optional): patch angle. Defaults to 0.
            patch_box (list, optional): patch box[x,y,hight,width]. Defaults to [0,0,60,30].
        """
        self.map_explorer = map_explorer
        self.vector_map = vector_map
        self.visual = visual
        self.patch_angle = patch_angle
        self.patch_box = patch_box
        self.tran_args=tran_args
        self.ann_name = 'ann'
        self.num_layer_elements = count_layer_element(geom_dict)

    def affine_transform_centerline(self):
        """Affine tranform a path: a center line w/ lanes and crosswalks in series"""
        for pat in ['rot_lan', 'sca_lan', 'shi_lan']:
            trans = getattr(self.tran_args, pat)
            if trans[0]:
                trans_dic = {}
                trans_dic['name'] = pat
                trans_dic['tran'] = trans
                self.del_centerline(aff=trans_dic)
                self.add_centerline(aff=trans_dic)
        
    def add_centerline(self, aff=None):
        """Add a path: a center line w/ lanes and crosswalks in series"""
        centerlines = self.geom_dict['centerline']

        if self.tran_args.diy:
            add_centerlines = geometry_manager(self.geom_dict, 'centerline', ['lane', 'ped_crossing', 'boundary'])
        else:
            if aff is not None:
                add_centerlines = self.delet_centerlines
            else:
                times = math.ceil(len(centerlines.keys()) * self.tran_args.add_lan[1])
                if times == 0:
                    return self.geom_dict

                add_centerlines = random_select_element(centerlines, times)

        for centerline_dic in add_centerlines:
            # Affine transformation parameter settings
            xoff = 0
            yoff = 0
            angle = 0
            origin_rot = 'center'
            xfact = 1
            yfact = 1
            origin_sca = 'center'
            
            if aff is None:
                xoff = random.choice([-3.5, 3.5])
                yoff = random.choice([-3.5, 3.5])
                angle = random.randint(-180, 180)
                origin_rot = (0,0)
                xfact = random.choice([-1, 1])
                yfact = random.choice([-1, 1])
                origin_sca = 'center'
            else:
                if aff['name'] == 'shi_lan':
                    xoff = random.uniform(aff['tran'][2][0][0], aff['tran'][2][0][1])
                    yoff = random.uniform(aff['tran'][2][1][0], aff['tran'][2][1][1])
                elif aff['name'] == 'rot_lan':
                    angle = random.randint(aff['tran'][2][0][0], aff['tran'][2][0][1])
                    origin_rot = aff['tran'][2][1]
                elif aff['name'] == 'sca_lan':
                    xfact = self.tran_args.rot_pat[2][0]
                    yfact = self.tran_args.rot_pat[2][1]
                    origin_sca = self.tran_args.sca_pat[2][2]
            
                        
            new_lane = centerline_dic['geom']
            new_lane = affine_transfer_4_add_centerline(new_lane, xoff, yoff, angle, origin_rot, xfact, yfact, origin_sca)
            centerlien_dic = layer_dict_generator(new_lane, source='new')
            self.geom_dict['centerline'][centerlien_dic['token']] = centerlien_dic
            
            center_lane = new_lane.buffer(1.8)
            new_lane_dic = layer_dict_generator(center_lane, source='centerline')
            self.geom_dict['lane'][new_lane_dic['token']] = new_lane_dic
            
            if 'lane_token' in centerline_dic:
                for lane_token in centerline_dic['lane_token']:
                    if lane_token in self.geom_dict['lane']:
                        lane_dic = self.geom_dict['lane'][lane_token]
                        if lane_dic['from'] != 'lane_connector':
                            new_lane = lane_dic['geom']
                            new_lane = affine_transfer_4_add_centerline(new_lane, xoff, yoff, angle, origin_rot, xfact, yfact, origin_sca)
                            new_lane_dic = layer_dict_generator(new_lane, source='new')
                            self.geom_dict['lane'][new_lane_dic['token']] = new_lane_dic
                    
            # if 'ped_crossing_token' in centerline_dic: #FIXME
            #     for lane_token in centerline_dic['ped_crossing_token']:
            #         lane_dic = self.geom_dict['ped_crossing'][lane_token]
            #         new_lane = lane_dic['geom']
            #         new_lane = affine_transfer_4_add_centerline(new_lane, xoff, yoff, angle, origin, xfact, yfact)
            #         new_lane_dic = layer_dict_generator(new_lane, source='new')
            #         self.geom_dict['ped_crossing'][new_lane_dic['token']] = new_lane_dic

    def del_centerline(self, aff=None):
        """Delet a path: a center line w/ lanes and crosswalks in series"""
        centerlines = self.geom_dict['centerline']
        road_segments = self.geom_dict['boundary']
        lanes = self.geom_dict['lane']
        lane_dividers = self.geom_dict['divider']
        ped_crossings = self.geom_dict['ped_crossing']
        agents = self.geom_dict['agent']
        
        if self.tran_args.diy:
            self.delet_centerlines = geometry_manager(self.geom_dict, 'centerline', ['lane', 'ped_crossing', 'boundary'])
            
            # for cl_dict in self.delet_centerlines:
            #     delet_centerline = centerlines.pop(cl_dict['token'])
        else:
            if aff is not None:
                times = math.ceil(len(centerlines.keys()) * aff['tran'][1])
            else:
                times = math.ceil(len(centerlines.keys()) * self.tran_args.del_lan[1])
            if times == 0:
                return self.geom_dict

            self.delet_centerlines = random_select_element(centerlines, times)
            # for key in self.delet_centerlines.keys():
            #     delet_centerline = centerlines.pop(key)

        # check if there are map layers are inuse after remove centerlines
        # remove a centerline need to also remove it's token from connected lanes and ped_crossings
        # if a lane or ped_crossing connected no centerline, it should be removed
        delet_lanes = []
        for del_cl in self.delet_centerlines:
            centerlines.pop(del_cl['token'])
            for lane_token in del_cl['lane_token']:
                try:
                    lanes[lane_token]['centerline_token'].remove(del_cl['token'])
                except:
                    pass
                if not len(lanes[lane_token]['centerline_token']):
                    delet_lanes.append(lanes.pop(lane_token))

            for ped_token in del_cl['ped_crossing_token']:
                ped_crossings[ped_token]['centerline_token'].remove(
                    del_cl['token'])
                if not len(ped_crossings[ped_token]['centerline_token']):
                    ped_crossings.pop(ped_token)

        # check if there are agents are inuse or need update after remove lane
        # remove a lane need to also remove connected agent and agent trajectory
        # if a agent exits but it's trajectory nolonger be used, update it
        for del_lane in delet_lanes:
            if 'agent_eco_token' in del_lane.keys():
                for agent_token in del_lane['agent_eco_token']:
                    del_lane['agent_token'].remove(agent_token)
                    agents.pop(agent_token)
            if 'agent_token' in del_lane.keys():    
                for ag_tra_token in del_lane['agent_token']:
                    if ag_tra_token in agents.keys():
                        agents[ag_tra_token]['lane_tokens'] = []
        
        del_agents = []    
        for ag_token, ag_val in agents.items():
            if not len(ag_val['lane_token']):
                if len(ag_val['eco_lane_token']):
                    centerline_candidates = lanes[ag_val['eco_lane_token'][0]]['centerline_token']
                    if len(centerline_candidates):
                        new_tra_token = random.choice(centerline_candidates)
                        ag_val['lane_token'] = centerlines[new_tra_token]['lane_tokens']
                    else:
                        del_agents.append(ag_token)

        for ag_token in del_agents:
            agents.pop(ag_token)
        
        # check if there are dividers are inuse
        delet_dividers = []
        for delet_lane in delet_lanes:
            for divider_token in ['left_lane_divider_token', 'right_lane_divider_token']:
                if divider_token in delet_lane:
                    delet_dividers.append(delet_lane[divider_token])

            # check if there are overlap with road_segment
            for road_dic in road_segments.values():
                new_road_seg = keep_non_intersecting_parts(
                    road_dic['geom'], delet_lane['geom'])
                if len(new_road_seg):
                    road_dic['geom'] = new_road_seg[0]

        for token in delet_dividers:
            if token in lane_dividers:
                lane_dividers.pop(token)

        # check unresonable: remove isolated element
        road_segments = check_isolated(road_segments, [centerlines])
        lane_dividers = check_isolated(lane_dividers, [road_segments, lanes])
        ped_crossings = check_isolated(ped_crossings, [road_segments, lanes], True)

    def affine_transform_patch(self):
        """affine transfor all the map elements"""
        if self.tran_args.rot_pat[0]:
            rot_p = [random.randint(self.tran_args.rot_pat[2][0][0], self.tran_args.rot_pat[2][0][1]), self.tran_args.rot_pat[2][1]]
        if self.tran_args.sca_pat[0]:
            sca_p = [self.tran_args.rot_pat[2][0], self.tran_args.rot_pat[2][1], self.tran_args.sca_pat[2][2]]
        if self.tran_args.shi_pat[0]:
            shi_p = [random.uniform(self.tran_args.shi_pat[2][0][0], self.tran_args.shi_pat[2][0][1]),
                     random.uniform(self.tran_args.shi_pat[2][1][0], self.tran_args.shi_pat[2][1][1])]
        
        for key in self.geom_dict.keys():
            if len(self.geom_dict[key]):
                for ins_v in self.geom_dict[key].values():
                    ins = ins_v['geom']
                    if self.tran_args.rot_pat[0]:ins = affinity.rotate(ins, rot_p[0], rot_p[1])
                    if self.tran_args.sca_pat[0]:
                        ins = affinity.scale(ins, sca_p[0], sca_p[1], sca_p[2])
                    if self.tran_args.shi_pat[0]:
                        ins = affinity.translate(ins, shi_p[0], shi_p[1])

                    ins_v['geom'] = ins
                    # geom = valid_geom(
                    #     ins, [0, 0, self.patch_box[2], self.patch_box[3]], 0)
                    
                    # if geom is None:
                    #     del self.geom_dict[key][ind]
                    #     # del correspondence_list[key][ind]
                    # else:
                    # if geom.geom_type == 'MultiLineString':
                    #     ins_v['geom'] = ops.linemerge(geom)
                    # else:
                    #     ins_v['geom'] = geom

    def adjust_lane_width(self):
        """Widen the boundaries of a path: the boundaries of all lanes and crosswalks connected by the center line"""
        move_distance = self.tran_args.wid_lan[2] / 2.0
        
        if self.tran_args.diy:
            # select_centerlines = geometry_manager(self.geom_dict, 'centerline', ['lane', 'ped_crossing', 'boundary'])
            pass #TODO
        else:
            times = math.ceil(len(self.geom_dict['centerline']) * self.tran_args.wid_lan[1])
            if times == 0:
                return self.geom_dict
            
            index_avaliable = [str(i) for i in range(len(self.geom_dict['centerline']))]
            
            centerline_dict = {}
            for ind, cl in enumerate(self.geom_dict['centerline']):
                centerline_dict[str(ind)] = cl
            
            for _ in range(times):
                if index_avaliable:
                    chosen_index = random.choice(index_avaliable)
                    index_avaliable.remove(chosen_index)
                else:
                    break
        
                centerline = centerline_dict[chosen_index]
                centerline_center = Point(centerline.centroid)
                for k, polylines in self.geom_dict.items():
                    moved_polylines = []
                    
                    if k != 'centerline':
                        for i, polyline in enumerate(polylines):
                            moved_polyline = move_geom(centerline_center, polyline, move_distance)
                            geom = valid_geom(moved_polyline, self.map_explorer, [0, 0, self.patch_box[2], self.patch_box[3]], 0)
                            if geom:
                                moved_polylines.append(geom)
                    else:
                        remove_index = []
                        for i, polyline in enumerate(centerline_dict.values()):
                            if str(i) in index_avaliable:
                                moved_polyline = move_geom(centerline_center, polyline, move_distance)
                                geom = valid_geom(moved_polyline, self.map_explorer, [0, 0, self.patch_box[2], self.patch_box[3]], 0)
                                if geom:
                                    moved_polylines.append(geom)
                                else:
                                    remove_index.append(str(i))
                        moved_polylines.append(centerline)
                        for id in remove_index:
                            index_avaliable.remove(id)
                            
                    self.geom_dict[k] = moved_polylines
            
    def delete_layers(self, map_vector_dict, correspondence_list, layer_name, args):
        times = math.ceil(self.num_layer_elements[layer_name] * args[1])
        for _ in range(times):
            if len(map_vector_dict[layer_name]):
                ind = random.randrange(len(map_vector_dict[layer_name]))
                del map_vector_dict[layer_name][ind]
                del correspondence_list[layer_name][ind]

        return map_vector_dict, correspondence_list

    def shift_layers(self, instance_list, correspondence_list, layer_name, args):
        times = math.floor(self.num_layer_elements[layer_name] * args[1])
        index_list = random.choices([i for i in range(self.num_layer_elements[layer_name])], k=times)
        r_xy = np.random.normal(0, 1, [times, 2])

        for ind in index_list:
            # rx = random.uniform(-1*args[2], args[2])
            # ry = math.sqrt(1 - pow(rx,2))
            rx = r_xy[ind][0]
            ry = r_xy[ind][1]
            geom = affinity.translate(instance_list[layer_name][ind], rx, ry)

            geom = valid_geom(geom, [0, 0, self.patch_box[2], self.patch_box[3]], 0)

            if geom is None:
                rx *= -1
                ry *= -1
                geom = affinity.translate(
                    instance_list[layer_name][ind], rx, ry)
                geom = valid_geom(geom, [0, 0, self.patch_box[2], self.patch_box[3]], 0)

                if geom is None:
                    del instance_list[layer_name][ind]
                    del correspondence_list[layer_name][ind]

                    continue

            if geom.geom_type == 'MultiLineString':
                instance_list[layer_name][ind] = ops.linemerge(geom)
            else:
                instance_list[layer_name][ind] = geom

        return instance_list, correspondence_list

    def shift_ped_crossing(self):
        if self.tran_args.diy:
            celected_geoms = geometry_manager(self.geom_dict, 'ped_crossing', ['lane', 'centerline', 'boundary'])
        else:
            times = math.ceil(len(self.geom_dict['ped_crossing'].keys()) * self.tran_args.shi_ped[1])
            if times == 0:
                return self.geom_dict

            celected_geoms = random_select_element(self.geom_dict['ped_crossing'], times)
            
        for ped_cro in celected_geoms:
            if ped_cro['geom'].geom_type == 'MultiPolygon':
                ped_cro['geom'] = unary_union(ped_cro['geom'])
                if ped_cro['geom'].geom_type == 'MultiPolygon':
                    print('ped_crossing is MultiPolygon and the shift is failed')
                    continue
            
            if 'centerline_token' in ped_cro:
                if len(ped_cro['centerline_token']):
                    along_cl_token = random.choice(ped_cro['centerline_token'])
                    along_cl = self.geom_dict['centerline'][along_cl_token]
                    if 'lane_token' in along_cl:
                        if len(along_cl['lane_token']):
                            moved_ped_forward, moved_ped_backward = move_polygon_along_polyline(ped_cro['geom'], along_cl['geom'], self.tran_args.shi_ped[2])
                            for moved_ped in moved_ped_forward, moved_ped_backward:
                                check = 0
                                for l_token in along_cl['lane_token']:
                                    lane = self.geom_dict['lane'][l_token]
                                    if moved_ped.intersects(lane['geom']):
                                        check = 1
                                        break
                                if check:
                                    self.geom_dict['ped_crossing'][ped_cro['token']]['geom'] = moved_ped
                                    break

    def shift_map(self):
        xoff = random.uniform(self.tran_args.shi_pat[2][0][0], self.tran_args.shi_pat[2][0][1])
        yoff = random.uniform(self.tran_args.shi_pat[2][1][0], self.tran_args.shi_pat[2][1][1])
        for layer, vect_list in self.vect_dict.items():
            if len(vect_list):
                for ind, vect in enumerate(vect_list):
                    new_point_list = []
                    for point in vect:
                        x = point[0] + xoff
                        y = point[1] + yoff
                        new_point_list.append([x, y])
                    
                    self.vect_dict[layer][ind] = np.array(new_point_list)
                    
    def rotate_map(self):
        angle = random.randint(self.tran_args.rot_pat[2][0])
        for layer, vect_list in self.vect_dict.items():
            if len(vect_list):
                for ind, vect in enumerate(vect_list):
                    new_point_list = []
                    for point in vect:
                        x = point[0] * math.cos(angle) - point[1] * math.sin(angle)
                        y = point[1] * math.sin(angle) + point[1] * math.cos(angle)
                        new_point_list.append([x, y])

                    self.vect_dict[layer][ind] = np.array(new_point_list)
    
    def flip_map(self):
        for layer, vect_list in self.vect_dict.items():
            if len(vect_list):
                for ind, vect in enumerate(vect_list):
                    new_point_list = []
                    for point in vect:
                        x = point[0] * self.tran_args.sca_pat[2][0]
                        y = point[1] * self.tran_args.sca_pat[2][1]
                        new_point_list.append([x, y])

                    self.vect_dict[layer][ind] = np.array(new_point_list)
        
    def zoom_layers(self, instance_list, correspondence_list, layer_name, args):
        times = math.floor(self.num_layer_elements[layer_name] * args[1])
        index_list = random.choices([i for i in range(self.num_layer_elements[layer_name])], k=times)

        new_ins_list = []
        new_cor_list = []
        for ind, ele in enumerate(instance_list[layer_name]):
            if ind in index_list:
                centroid = np.array(ele.centroid.coords)
                mv = centroid / abs(np.max(centroid)) * args[2]
                rx = mv[0][0]
                ry = mv[0][1]
                geom = affinity.translate(ele, rx, ry)

                geom = valid_geom(geom, [0, 0, self.patch_box[2], self.patch_box[3]], 0)

                if geom is None:
                    continue

                if geom.geom_type == 'MultiLineString':
                    new_ins_list.append(ops.linemerge(geom))
                    new_cor_list.append(correspondence_list[layer_name][ind])
                else:
                    new_ins_list.append(geom)
                    new_cor_list.append(correspondence_list[layer_name][ind])
            else:
                new_ins_list.append(ele)
                new_cor_list.append(correspondence_list[layer_name][ind])

        instance_list[layer_name] = new_ins_list
        correspondence_list[layer_name] = new_cor_list

        return instance_list, correspondence_list

    def _zoom_grid(self, param, zoom_are=None):
        nx, ny = int(self.patch_box[3]+1)+2, int(self.patch_box[2]+1)+2
        x = np.linspace(-int(self.patch_box[3]/2)-1, int(self.patch_box[3]/2)+1, nx)
        y = np.linspace(-int(self.patch_box[2]/2)-1, int(self.patch_box[2]/2)+1, ny)
        xv, yv = np.meshgrid(x, y, indexing='ij')

        xv = xv.reshape(33, 63, 1)
        yv = yv.reshape(33, 63, 1)
        xyv = np.concatenate((xv, yv), axis=2)
        xyv_max = np.reshape(np.max(abs(xyv), 2), (33, 63, 1))
        xyv_max = np.concatenate((xyv_max, xyv_max), axis=2)
        np.seterr(invalid='ignore')
        xy_mv = np.multiply(np.divide(xyv, xyv_max), param)

        if zoom_are is not None:
            xmin = math.floor(zoom_are[0]) + 16
            ymin = math.floor(zoom_are[1]) + 31
            xmax = math.ceil(zoom_are[2]) + 16
            ymax = math.ceil(zoom_are[3]) + 31

            xyv[xmin:xmax+1, ymin:ymax+1, :] = xyv[xmin:xmax+1,ymin:ymax+1, :] + xy_mv[xmin:xmax+1, ymin:ymax+1, :]
        else:
            xyv += xy_mv

        return xyv[:, :, 0], xyv[:, :, 1]

    def _warping(self, ins, xv, yv):
        new_point_list = []
        for point in ins:
            x = point[0]
            y = point[1]

            if math.isnan(x) or math.isnan(y):
                continue
            
            # canonical top left
            x_floor = math.floor(x)
            y_floor = math.floor(y)

            # Check upper or lower triangle
            x_res = x - x_floor
            y_res = y - y_floor
            upper = (x_res+y_res) <= 1.0

            # transfer x_floor coord[-15,15] to x_floor ind[0,32] fro grid
            x_floor += 16
            y_floor += 31

            if upper:
                # Get anchor
                x_anc = xv[x_floor, y_floor]
                y_anc = yv[x_floor, y_floor]

                # Get basis
                x_basis_x = xv[x_floor+1, y_floor] - x_anc
                x_basis_y = yv[x_floor+1, y_floor] - y_anc

                y_basis_x = xv[x_floor, y_floor+1] - x_anc
                y_basis_y = yv[x_floor, y_floor+1] - y_anc
            else:
                # Get anchor
                x_anc = xv[x_floor+1, y_floor+1]
                y_anc = yv[x_floor+1, y_floor+1]

                # Get basis
                x_basis_x = xv[x_floor, y_floor+1] - x_anc
                x_basis_y = yv[x_floor, y_floor+1] - y_anc

                y_basis_x = xv[x_floor+1, y_floor] - x_anc
                y_basis_y = yv[x_floor+1, y_floor] - y_anc
                x_res = 1-x_res
                y_res = 1-y_res

            # Get new coordinate in warped mesh
            x_warp = x_anc + x_basis_x * x_res + y_basis_x * y_res
            y_warp = y_anc + x_basis_y * x_res + y_basis_y * y_res

            new_point_list.append((x_warp, y_warp))

        return np.array(new_point_list)

    def zoom_patch_by_layers(self, layer_name):
        times = math.floor(self.num_layer_elements[layer_name] * self.tran_args.wid_lan[1])
        index_list = random.choices([i for i in range(self.num_layer_elements[layer_name])], k=times)

        for ind, ele in enumerate(self.vect_dict[layer_name]):
            if ind in index_list:
                widen_area = (ele[:, 0].min(), ele[:, 1].min(),ele[:, 0].max(), ele[:, 1].max())
                g_xv, g_yv = self._zoom_grid(self.patch_box, self.tran_args.wid_lan[2], widen_area)

                for key in self.vect_dict.keys():
                    if len(self.vect_dict[key]):
                        for ind, ins in enumerate(self.vect_dict[key]):
                            ins = fix_corner(ins, [0, 0, self.patch_box[2], self.patch_box[3]])
                            self.vect_dict[key][ind] = self._warping(ins, g_xv, g_yv)

    def _con(self, ins, xlim, ylim, randx, randy, randr):
        new_point_list = []
        for point in ins:

            j = point[0]
            i = point[1]

            offset_x = randx * math.sin(2 * 3.14 * i / 150)
            offset_y = randy * math.cos(2 * 3.14 * j / 150)
            offset_x += randr * math.sin(2 * 3.14 * i / (2*(xlim[1]-xlim[0])))

            new_point = []
            if j in xlim:
                new_point.append(j)
            else:
                if xlim[0] < j+offset_x < xlim[1]:
                    new_point.append(j+offset_x)
                # elif xlim[1] < j+offset_x:
                #     new_point.append(xlim[1])
                # elif j+offset_x < xlim[0]:
                #     new_point.append(xlim[0])
                else:
                    continue

            if i in ylim:
                new_point.append(i)
            else:
                if ylim[0] < i+offset_y < ylim[1]:
                    new_point.append(i+offset_y)
                # elif ylim[1] < i+offset_y:
                #     new_point.append(ylim[1])
                # elif i+offset_y < ylim[0]:
                #     new_point.append(ylim[0])
                else:
                    continue

            new_point_list.append(np.array(new_point))
        
        return np.array(new_point_list)

    def difromate_map(self):
        """Trigonometric warping map patches"""
        # Vertical distortion amplitude random maximum range int
        v = self.tran_args.def_pat_tri[2][0]
        # Horizontal distortion amplitude random maximum range int
        h = self.tran_args.def_pat_tri[2][1]
        i = self.tran_args.def_pat_tri[2][2]  # Inclination amplitude [-Max_r Max_r] int

        xlim = [-self.patch_box[3]/2, self.patch_box[3]/2]
        ylim = [-self.patch_box[2]/2, self.patch_box[2]/2]

        for vect_list in self.vect_dict.values():
            if len(vect_list):
                invalide_vect_ind = []
                for ind, ins in enumerate(vect_list):
                    new_vect = self._con(ins, xlim, ylim, v, h, i)
                    if new_vect.any():
                        vect_list[ind] = new_vect
                    else:
                        invalide_vect_ind.append(ind)
                
                vect_list = [item for i, item in enumerate(vect_list) if i not in invalide_vect_ind]
   
    def _gaussian_grid(self, def_args):
        nx, ny = int(self.patch_box[3]+1)+2, int(self.patch_box[2]+1)+2
        x = np.linspace(-int(self.patch_box[3]/2)-1, int(self.patch_box[3]/2)+1, nx)
        y = np.linspace(-int(self.patch_box[2]/2)-1, int(self.patch_box[2]/2)+1, ny)
        xv, yv = np.meshgrid(x, y, indexing='ij')
        g_xv = xv + \
            np.random.normal(def_args[2][0], def_args[2][1], size=[nx, ny])
        g_yv = yv + \
            np.random.normal(def_args[2][0], def_args[2][1], size=[nx, ny])
        g_xv[:2, :] = xv[:2, :]
        g_xv[nx-2:, :] = xv[nx-2:, :]
        g_yv[:, :2] = yv[:, :2]
        g_yv[:, ny-2:] = yv[:, ny-2:]

        return g_xv, g_yv

    def guassian_warping(self):
        """gaussian warping map patches"""
        g_xv, g_yv = self._gaussian_grid(self.tran_args.def_pat_gau)

        for key in self.vect_dict.keys():
            if len(self.vect_dict[key]):
                for ind, ins in enumerate(self.vect_dict[key]):
                    ins = fix_corner(ins, [0, 0, self.patch_box[2], self.patch_box[3]])
                    self.vect_dict[key][ind] = self._warping(ins, g_xv, g_yv)

    def guassian_noise(self):
        """add gaussian noise to map patches"""
        for key in self.vect_dict.keys():
            if len(self.vect_dict[key]):
                for ind, ins in enumerate(self.vect_dict[key]):
                    g_nois = np.random.normal(
                        self.tran_args.noi_pat_gau[2][0], self.tran_args.noi_pat_gau[2][1], ins.shape)
                    self.vect_dict[key][ind] += g_nois

    def truncate_and_save(self, map_type, map_name = 'inter_pt'):
        """Visualizing the intermediate steps of perturbations

        Args:
            map_type (str): geometry or vectory
            map_name (str, optional): output name. Defaults to 'inter_pt'.
        """
        if self.tran_args.truncate:
            if map_type == 'geom':
                ## transfer geom to ready-show vectorized geom 
                trans_dic = self.vector_map.gen_vectorized_samples(self.geom_dict)
                ## transform vectorized geom to numpy array
                map_vect_dic = geom_to_np(trans_dic)
            elif map_type == 'geom_ins':
                map_vect_dic = geom_to_np(self.geom_dict)
            elif map_type == 'vect':
                map_vect_dic = self.vect_dict
                
            trans_np_dict_4_vis = np_to_geom(map_vect_dic)
            trans_np_dict_4_vis = geom_to_np(trans_np_dict_4_vis, inter=True)
            self.visual.vis_contours(trans_np_dict_4_vis, self.ann_name + '_truncate_' + map_name)
            
    def perturb_vect_map(self):
        """Perturb the map vectory layer: image algorithms acting on numpy arrays"""
        self.vect_dict = geom_to_np(self.geom_dict, inter=True)

        if self.tran_args.shi_pat[0]:
            self.shift_map()
            self.truncate_and_save('vect', '6_shift_map')
            
        if self.tran_args.rot_pat[0]:
            self.rotate_map()
            self.truncate_and_save('vect', '7_rotate_map')
            
        if self.tran_args.sca_pat[0]:
            self.flip_map() #TOFIX
            self.truncate_and_save('vect', '8_flip_map')
        
        if self.tran_args.def_pat_tri[0]:
            self.difromate_map()
            self.truncate_and_save('vect', '9_warping_tri')

        if self.tran_args.def_pat_gau[0]:
            self.guassian_warping()
            self.truncate_and_save('vect', '10_warping_gua')

        if self.tran_args.noi_pat_gau[0]:
            self.guassian_noise()
            self.truncate_and_save('vect', '11_noiseing')

    def perturb_geom_layer(self, geom_dict):
        """Perturb the map geometry layer: 'centerline'-based perturbations"""
        self.geom_dict = geom_dict
        
        if self.tran_args.del_lan[0]:
            self.del_centerline()
            self.truncate_and_save('geom', '1_delet_lane')

        if self.tran_args.add_lan[0]:
            self.add_centerline()
            self.truncate_and_save('geom', '2_add_lane')
        
        if any(getattr(self.tran_args, pat)[0] for pat in ['rot_lan', 'sca_lan', 'shi_lan']):
            self.affine_transform_centerline()
            self.truncate_and_save('geom', '3_affine_transform_lane')
        
        if self.tran_args.shi_ped[0]:
            self.shift_ped_crossing()
            self.truncate_and_save('geom', '4_shift_ped_crossing')
        
        # if any(getattr(self.tran_args, pat)[0] for pat in ['rot_pat', 'sca_pat', 'shi_pat']): #FIXME
        #     self.affine_transform_patch()
        #     self.truncate_and_save('geom', '5_affine_transform_map')
        
        self.geom_dict_for_json = copy.deepcopy(self.geom_dict)
        self.geom_dict = self.vector_map.gen_vectorized_samples(self.geom_dict)
        if self.tran_args.wid_lan[0]:
            self.adjust_lane_width()
            self.truncate_and_save('geom_ins', '5_adjust_lane')

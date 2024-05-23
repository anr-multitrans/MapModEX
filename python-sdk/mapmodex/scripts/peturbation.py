import math
import random
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from shapely import affinity, ops
from shapely.geometry import Point, Polygon, LineString

from ..utils import *


class PerturbParameters():
    def __init__(self, pt_name=None,
                 # [switch, proportion, parameter]
                 # ped_crossing perturbation
                 del_ped=[0, 0, None],  # delete ped_crossing
                 # shift ped_crossing in its road_segment, shifted by offsets along each dimension[x, y]
                 shi_ped=[0, 0, [0, 0]],
                 add_ped=[0, 0, None],  # add ped_crossing in a road_segment
                 # dividers perturbation
                 del_div=[0, 0, None],  # delete divider
                 # shift divider, shifted by offsets along each dimension[x, y]
                 shi_div=[0, 0, [0, 0]],
                 add_div=[0, 0, None],  # add divider TODO
                 del_lan=[0, 0, None],
                 add_lan=[0,0,None],
                 wid_lan=[0, 0, None],
                 # boundray perturabtion
                 del_bou=[0, 0, None],  # delete lane
                 # shift lane, shifted by offsets along each dimension[x, y]
                 shi_bou=[0, 0, [0, 0]],
                 add_bou=[0, 0, None],  # add boundray TODO
                 wid_bou=[0, 0, None],
                 # patch perturbation
                 aff_tra_pat=[0, None, [1, 0, 0, 1, 0, 0]],  # affine_transform: [a, b, d, e, xoff, yoff], x' = a * x + b * y + xoff, y' = d * x + e * y + yoff
                 rot_pat=[0, None, [0, [0, 0]]],  # rotate the patch
                 sca_pat=[0, None, [1, 1]],  # scale the patch
                 ske_pat=[0, None, [0, 0, (0, 0)]],  # skew the patch
                 shi_pat=[0, None, [0, 0]],  # translate: shift the patch
                 # Horizontal, Vertical, and Inclination distortion amplitude
                 def_pat_tri=[0, None, [0, 0, 0]],
                 # gaussian mean and standard deviation
                 def_pat_gau=[0, None, [0, 1]],
                 # gaussian mean and standard deviation
                 noi_pat_gau=[0, None, [0, 1]],
                 # Interpolation
                 int_num=0,
                 int_ord='before',  # before the perturbation or after it
                 int_sav=False):  # save the interpolated instances

        self.pt_name = pt_name
        
        self.del_ped = del_ped
        self.shi_ped = shi_ped
        self.add_ped = add_ped
        self.del_div = del_div
        self.shi_div = shi_div
        self.add_div = add_div
        self.del_lan = del_lan
        self.wid_lan = wid_lan
        self.add_lan = add_lan
        self.del_bou = del_bou
        self.shi_bou = shi_bou
        self.add_bou = add_bou
        self.wid_bou = wid_bou
        self.aff_tra_pat = aff_tra_pat
        self.rot_pat = rot_pat
        self.sca_pat = sca_pat
        self.ske_pat = ske_pat
        self.shi_pat = shi_pat
        self.def_pat_tri = def_pat_tri
        self.def_pat_gau = def_pat_gau
        self.noi_pat_gau = noi_pat_gau

        self.int_num = int_num
        self.int_ord = int_ord
        self.int_sav = int_sav

    def update_attribute(self, attr_name, new_value):
        if hasattr(self, attr_name):
            setattr(self, attr_name, new_value)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr_name}'")     
    

class MapTransform:
    def __init__(self, map_explorer: NuScenesMapExplorer, layer_names=['ped_crossing', 'road_segment', 'road_block']):
        self.map_explorer = map_explorer
        self.layer_names = layer_names   # 'road_divider' are lines
        self.patch_angle = 0

    def transfor_patch(self, instance_list, args):
        # if args.aff_tra_pat[0]:
        #     ins = affinity.affine_transform(
        #         ins, args.aff_tra_pat[2])
        if args.rot_pat[0]:
            rot_p = [random.randint(args.rot_pat[2][0][0], args.rot_pat[2][0][1]), args.rot_pat[2][1]]
            # ins = affinity.rotate(ins, random.randint(args.rot_pat[2][0][0], args.rot_pat[2][0][1]), args.rot_pat[2][1])
        # if args.sca_pat[0]:
        #     ins = affinity.scale(
        #         ins, args.sca_pat[2][0], args.sca_pat[2][1])
        # if args.ske_pat[0]:
        #     ins = affinity.skew(
        #         ins, args.ske_pat[2][0], args.ske_pat[2][1], args.ske_pat[2][2])
        if args.shi_pat[0]:
            shi_p = [random.uniform(args.shi_pat[2][0][0], args.shi_pat[2][0][1]), random.uniform(args.shi_pat[2][1][0], args.shi_pat[2][1][1])]
            # ins = affinity.translate(ins, random.uniform(args.shi_pat[2][0][0], args.shi_pat[2][0][1]), random.uniform(args.shi_pat[2][1][0], args.shi_pat[2][1][1]))
        
        for key in instance_list.keys():
            if len(instance_list[key]):
                for ins_v in instance_list[key].values():
                    ins = ins_v['geom']
                    
                    if args.aff_tra_pat[0]:
                        ins = affinity.affine_transform(
                            ins, args.aff_tra_pat[2])
                    if args.rot_pat[0]:
                        ins = affinity.rotate(ins, rot_p[0], rot_p[1])
                    if args.sca_pat[0]:
                        ins = affinity.scale(
                            ins, args.sca_pat[2][0], args.sca_pat[2][1])
                    if args.ske_pat[0]:
                        ins = affinity.skew(
                            ins, args.ske_pat[2][0], args.ske_pat[2][1], args.ske_pat[2][2])
                    if args.shi_pat[0]:
                        ins = affinity.translate(ins, shi_p[0], shi_p[1])

                    ins_v['geom'] = ins
                    # geom = valid_geom(
                    #     ins, [0, 0, patch_box[2], patch_box[3]], 0)
                    
                    # if geom is None:
                    #     del instance_list[key][ind]
                    #     # del correspondence_list[key][ind]
                    # else:
                    # if geom.geom_type == 'MultiLineString':
                    #     ins_v['geom'] = ops.linemerge(geom)
                    # else:
                    #     ins_v['geom'] = geom

        return instance_list

    def creat_ped_polygon(self, road_segment_token=None):

        min_x, min_y, max_x, max_y = self.map_explorer.map_api.get_bounds(
            'road_segment', road_segment_token)

        x_range = max_x - min_x
        y_range = max_y - min_y

        if max([x_range, y_range]) <= 4:
            new_polygon = self.map_explorer.map_api.extract_polygon(
                self.map_explorer.map_api.get('road_segment', road_segment_token)['polygon_token'])
        else:
            if x_range > y_range:
                rand = random.uniform(min_x, max_x - 4)
                left_bottom = Point([rand, min_y])
                left_top = Point([rand, max_y])
                right_bottom = Point([rand + 4, min_y])
                right_top = Point([rand + 4, max_y])
            else:
                rand = random.uniform(min_y, max_y - 4)
                left_bottom = Point([min_x, rand])
                left_top = Point([min_x, rand + 4])
                right_bottom = Point([max_x, rand])
                right_top = Point([max_x, rand + 4])

            new_polygon = Polygon(
                [left_top, left_bottom, right_bottom, right_top])

        return new_polygon

    def creat_boundray(self, bd, patch_box):
        # Get the center point of the boundary and draw a circle with this center point
        center_point = bd.interpolate(bd.length / 2)
        c = center_point.buffer(3.7).boundary
        # Get the intersection point of the boundary and the circle
        bi = c.intersection(bd)

        # Determine the starting point of the new boundary based on the number of intersection points
        if bi.geom_type == 'MultiPoint':
            pt_1 = np.array(bi.geoms[0].coords, float)
            pt_2 = np.array(bi.geoms[-1].coords, float)
        elif bi.geom_type == 'Point':
            pt_1 = np.array(bi.coords, float)
            pt_2 = None
        else:
            return []

        limit_1 = min((np.array([[15, 30]]) - abs(pt_1)) /
                      abs(np.array(center_point.coords)))
        pt_bd_1 = list(list(pt_1 + limit_1*np.array(center_point.coords))[0])
        new_b_1 = LineString([pt_bd_1, list(list(pt_1)[0])])

        if pt_2 is not None:
            limit_2 = min(
                (np.array([[15, 30]]) - abs(pt_2)) / abs(np.array(center_point.coords)))
            pt_bd_2 = list(
                list(pt_2 + limit_2*np.array(center_point.coords))[0])
            new_b_2 = LineString([pt_bd_2, list(list(pt_2)[0])])

            return [new_b_1, new_b_2]

        return [new_b_1]

    def add_centerline(self, map_geom_dict, tran_args):   #TODO
        centerlines = map_geom_dict['centerline']

        times = math.ceil(len(centerlines.keys()) * tran_args[1])
        if times == 0:
            times = 1

        new_deviders = []
        new_centerlines = []
        for _ in range(times):
            centerline = centerlines[random.choice([k for k in centerlines.keys()])]['geom']
            if centerline.geom_type == 'MultiLineString':
                continue
            # affinity tranform
            centerline = affinity.translate(centerline, random.choice(
                [-3.5, 3.5]), random.choice([-3.5, 3.5])) #shift
            centerline = affinity.rotate(centerline, random.randint(-180, 180)) #rotate
            centerline = affinity.scale(centerline, random.choice([-1, 1]), random.choice([-1, 1])) #flip
            centerlien_dic = layer_dict_generator(centerline, source='centerline')
            map_geom_dict['centerline'][centerlien_dic['token']] = centerlien_dic
            
            center_lane = centerline.buffer(1.75) #TODO use original lanes
            if center_lane.geom_type == 'MultiPolygon':
                continue
            lane_dic = layer_dict_generator(center_lane, source='new')
            map_geom_dict['lane'][lane_dic['token']] = lane_dic

            # for divider in map_geom_dict['divider']:
            #     if not divider.is_empty:
            #         divider = interpolate(divider)
            #         new_deviders += keep_non_intersecting_parts(
            #             divider, center_lane)

            # new_centerlines.append(centerline)
            
            # map_geom_dict['divider'] = new_deviders

        # map_geom_dict['centerline'] += new_centerlines

        # map_geom_dict['centerline'] = union_line(map_geom_dict['centerline'])

        return map_geom_dict

    def widden_lane(self, map_geom_org_dic,  tran_args):
        centerlines = map_geom_org_dic['centerline']
        road_segments = map_geom_org_dic['boundary']
        lanes = map_geom_org_dic['lane']
        # lane_dividers = map_geom_org_dic['divider']
        ped_crossings = map_geom_org_dic['ped_crossing']
        # agents = map_geom_org_dic['agent']
        
        times = math.ceil(len(centerlines.keys()) * tran_args[1])
        if times == 0:
            return map_geom_org_dic

        select_centerlines = random_unique_items(centerlines, times)
        for cl in select_centerlines.values():
            road_segments = move_polygons(cl['geom'], road_segments, 2)
            lanes = move_polygons(cl['geom'], lanes, 2)
            centerlines = move_polygons(cl['geom'], centerlines, 2, except_token = [cl['token']])
            ped_crossings = move_polygons(cl['geom'], ped_crossings, 2)
            # agents = move_polygons(cl['geom'], agents, 2)
            
        return map_geom_org_dic
            
    def del_centerline(self, map_geom_org_dic,  tran_args):
        centerlines = map_geom_org_dic['centerline']
        road_segments = map_geom_org_dic['boundary']
        lanes = map_geom_org_dic['lane']
        lane_dividers = map_geom_org_dic['divider']
        ped_crossings = map_geom_org_dic['ped_crossing']
        agents = map_geom_org_dic['agent']

        times = math.ceil(len(centerlines.keys()) * tran_args[1])
        if times == 0:
            return map_geom_org_dic

        delet_centerlines = random_unique_items(centerlines, times)
        for key in delet_centerlines.keys():
            delet_centerline = centerlines.pop(key)

        # check if there are map layers are inuse after remove centerlines
        # remove a centerline need to also remove it's token from connected lanes and ped_crossings
        # if a lane or ped_crossing connected no centerline, it should be removed
        delet_lanes = []
        for del_cl in delet_centerlines.values():
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
        # generate divider later #TODO
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
        ped_crossings = check_isolated(
            ped_crossings, [road_segments, lanes], True)

        return map_geom_org_dic

    def delete_layers(self, instance_list, correspondence_list, len_dict, layer_name, args):
        times = math.ceil(len_dict[layer_name] * args[1])
        for _ in range(times):
            if len(instance_list[layer_name]):
                ind = random.randrange(len(instance_list[layer_name]))
                del instance_list[layer_name][ind]
                del correspondence_list[layer_name][ind]

        return instance_list, correspondence_list

    def shift_layers(self, instance_list, correspondence_list, len_dict, layer_name, args, patch_box):
        times = math.floor(len_dict[layer_name] * args[1])
        index_list = random.choices(
            [i for i in range(len_dict[layer_name])], k=times)
        r_xy = np.random.normal(0, 1, [times, 2])

        for ind in index_list:
            # rx = random.uniform(-1*args[2], args[2])
            # ry = math.sqrt(1 - pow(rx,2))
            rx = r_xy[ind][0]
            ry = r_xy[ind][1]
            geom = affinity.translate(
                instance_list[layer_name][ind], rx, ry)

            geom = valid_geom(geom, [0, 0, patch_box[2], patch_box[3]], 0)

            if geom is None:
                rx *= -1
                ry *= -1
                geom = affinity.translate(
                    instance_list[layer_name][ind], rx, ry)
                geom = valid_geom(geom, [0, 0, patch_box[2], patch_box[3]], 0)

                if geom is None:
                    del instance_list[layer_name][ind]
                    del correspondence_list[layer_name][ind]

                    continue

            if geom.geom_type == 'MultiLineString':
                instance_list[layer_name][ind] = ops.linemerge(geom)
            else:
                instance_list[layer_name][ind] = geom

        return instance_list, correspondence_list

    def zoom_layers(self, instance_list, correspondence_list, len_dict, layer_name, args, patch_box):
        times = math.floor(len_dict[layer_name] * args[1])
        index_list = random.choices(
            [i for i in range(len_dict[layer_name])], k=times)

        new_ins_list = []
        new_cor_list = []
        for ind, ele in enumerate(instance_list[layer_name]):
            if ind in index_list:
                centroid = np.array(ele.centroid.coords)
                mv = centroid / abs(np.max(centroid)) * args[2]
                rx = mv[0][0]
                ry = mv[0][1]
                geom = affinity.translate(ele, rx, ry)

                geom = valid_geom(geom, [0, 0, patch_box[2], patch_box[3]], 0)

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

    def zoom_grid(self, patch_box, def_args, zoom_are=None):
        nx, ny = int(patch_box[3]+1)+2, int(patch_box[2]+1)+2
        x = np.linspace(-int(patch_box[3]/2)-1, int(patch_box[3]/2)+1, nx)
        y = np.linspace(-int(patch_box[2]/2)-1, int(patch_box[2]/2)+1, ny)
        xv, yv = np.meshgrid(x, y, indexing='ij')

        xv = xv.reshape(33, 63, 1)
        yv = yv.reshape(33, 63, 1)
        xyv = np.concatenate((xv, yv), axis=2)
        xyv_max = np.reshape(np.max(abs(xyv), 2), (33, 63, 1))
        xyv_max = np.concatenate((xyv_max, xyv_max), axis=2)
        np.seterr(invalid='ignore')
        xy_mv = np.multiply(np.divide(xyv, xyv_max), def_args[2])

        if zoom_are is not None:
            xmin = math.floor(zoom_are[0]) + 16
            ymin = math.floor(zoom_are[1]) + 31
            xmax = math.ceil(zoom_are[2]) + 16
            ymax = math.ceil(zoom_are[3]) + 31

            xyv[xmin:xmax+1, ymin:ymax+1, :] = xyv[xmin:xmax+1,
                                                   ymin:ymax+1, :] + xy_mv[xmin:xmax+1, ymin:ymax+1, :]
        else:
            xyv += xy_mv

        return xyv[:, :, 0], xyv[:, :, 1]

    def warping(self, ins, xv, yv):
        new_point_list = []
        for point in ins:
            x = point[0]
            y = point[1]

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

    def zoom_patch_by_layers(self, map_ins_dict, len_dict, layer_name, args, patch_box):
        times = math.floor(len_dict[layer_name] * args[1])
        index_list = random.choices(
            [i for i in range(len_dict[layer_name])], k=times)

        for ind, ele in enumerate(map_ins_dict[layer_name]):
            if ind in index_list:
                widen_area = (ele[:, 0].min(), ele[:, 1].min(),
                              ele[:, 0].max(), ele[:, 1].max())
                g_xv, g_yv = self.zoom_grid(patch_box, args, widen_area)

                for key in map_ins_dict.keys():
                    if len(map_ins_dict[key]):
                        for ind, ins in enumerate(map_ins_dict[key]):
                            ins = fix_corner(
                                ins, [0, 0, patch_box[2], patch_box[3]])
                            map_ins_dict[key][ind] = self.warping(
                                ins, g_xv, g_yv)

        return map_ins_dict

    def add_layers(self, instance_list, correspondence_list, len_dict, layer_name, args, patch_box, patch_angle):
        times = math.ceil(len_dict[layer_name] * args[1])

        if layer_name == 'ped_crossing':
            if times == 0:
                times = 1

            patch_coords = self.patch_box_2_coords(patch_box)
            road_seg_records = self.map_explorer.map_api.get_records_in_patch(
                patch_coords, ['road_segment'])['road_segment']
            if len(road_seg_records):
                for _ in range(times):
                    new_geom_v = None
                    counter = 0
                    while new_geom_v is None:
                        counter += 1
                        if counter > 100:
                            print("this is going nowhere")
                            break
                        new_geom = self.creat_ped_polygon(
                            random.choice(road_seg_records))
                        new_geom_v = valid_geom(
                            new_geom, patch_box, patch_angle)
                        if new_geom_v is not None:
                            if new_geom_v.boundary.geom_type == 'MultiLineString':
                                instance_list[layer_name].append(
                                    ops.linemerge(new_geom_v.boundary))
                            else:
                                instance_list[layer_name].append(
                                    new_geom_v.boundary)

                            correspondence_list[layer_name].append(-1)

        if layer_name == 'boundary' and times:
            bd_index = [i for i in range(len(instance_list[layer_name]))]
            picked_boundary = random.sample(bd_index, times)
            for bd_id in picked_boundary:
                corr_ind = correspondence_list[layer_name][bd_id]
                new_bd = self.creat_boundray(
                    instance_list[layer_name][bd_id], patch_box)

                v_new_bd = []
                v_new_corr = []
                for bd in new_bd:
                    v_bd = valid_geom(
                        bd, [0, 0, patch_box[2], patch_box[3]], 0)
                    if v_bd is not None:
                        v_new_bd.append(v_bd)
                        v_new_corr.append(corr_ind)

                # instance_list[layer_name][bd_id:bd_id+1] = v_new_bd
                # correspondence_list[layer_name][bd_id:bd_id+1] = v_new_corr
                instance_list[layer_name][bd_id:bd_id] = v_new_bd
                correspondence_list[layer_name][bd_id:bd_id] = v_new_corr

        return instance_list, correspondence_list

    def con(self, ins, xlim, ylim, randx, randy, randr):
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
                elif xlim[1] < j+offset_x:
                    new_point.append(xlim[1])
                elif j+offset_x < xlim[0]:
                    new_point.append(xlim[0])

            if i in ylim:
                new_point.append(i)
            else:
                if ylim[0] < i+offset_y < ylim[1]:
                    new_point.append(i+offset_y)
                elif ylim[1] < i+offset_y:
                    new_point.append(ylim[1])
                elif i+offset_y < ylim[0]:
                    new_point.append(ylim[0])

            new_point_list.append(np.array(new_point))

        new_ins = np.array(new_point_list)
        return new_ins

    def difromate_map(self, map_ins_dict, def_args, patch_box):
        # Vertical distortion amplitude random maximum range int
        v = def_args[2][0]
        # Horizontal distortion amplitude random maximum range int
        h = def_args[2][1]
        i = def_args[2][2]  # Inclination amplitude [-Max_r Max_r] int

        xlim = [-patch_box[3]/2, patch_box[3]/2]
        ylim = [-patch_box[2]/2, patch_box[2]/2]

        for key in map_ins_dict.keys():
            if len(map_ins_dict[key]):
                for ind, ins in enumerate(map_ins_dict[key]):
                    map_ins_dict[key][ind] = self.con(
                        ins, xlim, ylim, v, h, i)

        return map_ins_dict

    def gaussian_grid(self, patch_box, def_args):
        nx, ny = int(patch_box[3]+1)+2, int(patch_box[2]+1)+2
        x = np.linspace(-int(patch_box[3]/2)-1, int(patch_box[3]/2)+1, nx)
        y = np.linspace(-int(patch_box[2]/2)-1, int(patch_box[2]/2)+1, ny)
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

    def guassian_warping(self, map_ins_dict, def_args, patch_box):
        g_xv, g_yv = self.gaussian_grid(patch_box, def_args)

        for key in map_ins_dict.keys():
            if len(map_ins_dict[key]):
                for ind, ins in enumerate(map_ins_dict[key]):
                    ins = fix_corner(ins, [0, 0, patch_box[2], patch_box[3]])
                    map_ins_dict[key][ind] = self.warping(ins, g_xv, g_yv)

        return map_ins_dict

    def guassian_noise(self, map_ins_dict, def_args):
        for key in map_ins_dict.keys():
            if len(map_ins_dict[key]):
                for ind, ins in enumerate(map_ins_dict[key]):
                    g_nois = np.random.normal(
                        def_args[2][0], def_args[2][1], ins.shape)
                    map_ins_dict[key][ind] += g_nois

        return map_ins_dict

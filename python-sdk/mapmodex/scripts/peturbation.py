import math
import random
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from shapely import affinity, ops
from shapely.geometry import Point

from ..utils import *


class PerturbParameters():
    """class for perturbation parameter setting"""
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
                 
                 # boundray perturbation
                 wid_bou=[0, 0, 0], #[switch, proportion, distance], the widening distance should be within a reasonable range (-2,15)
                 
                 # patch perturbation
                 rot_pat=[0, None, [[0,0], [0, 0]]],  #[switch, proportion, [angle_range, origin]], the origin should be the center coordinates of the patch_box.
                 sca_pat=[0, None, [1, 1, (0, 0)]],  #[switch, proportion, [xfact, yfact, origin]], xfact and yfact should be 1 or -1 to perform a flip.
                 shi_pat=[0, None, [[0,0], [0,0]]],  #[switch, proportion, [xoff_range, yoff_range]]
                 
                 def_pat_tri=[0, None, [0, 0, 0]], #[switch, proportion, [xamp, yamp, ramp]], Horizontal, Vertical, and Inclination distortion amplitude
                 def_pat_gau=[0, None, [0, 1]], # [switch, proportion, [mean, standard]]
                 noi_pat_gau=[0, None, [0, 1]], # [switch, proportion, [mean, standard]]
                 
                 ## other perturbation setting
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
            new_value (_type_): variable value

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
        """ initialization

        Args:
            map_explorer (NuScenesMapExplorer): Dataset class in the nuScenes map dataset explorer.
            geom_dict (dict): geometry map layers
            vector_map (VectorizedMap, optional): class of vectorize geometry map. Defaults to None.
            visual (RenderMap, optional): class in map visualization. Defaults to None.
            tran_args (dict, optional): perturbation parameters. Defaults to None.
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

    def _creat_ped_polygon(self, road_segment):
        """Create a crosswalk in a roadblock

        Args:
            road_segment (Polygon): Geometry of roadblocks

        Returns:
            Polygon: The geometry of a crosswalk
        """
        min_x, min_y, max_x, max_y = road_segment.bounds

        x_range = max_x - min_x
        y_range = max_y - min_y

        if max([x_range, y_range]) <= 4:
            new_polygon = road_segment
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

            new_polygon = Polygon([left_top, left_bottom, right_bottom, right_top])

        return new_polygon

    def _warping(self, ins, xv, yv):
        """mesh-based coordinate transformation

        Args:
            ins (np.array): coordinates
            xv (_type_): The x coordinate of the mesh
            yv (_type_): The y coordinate of the mesh

        Returns:
            np.array: new coordinate in the warped mesh
        """
        new_point_list = []
        for point in ins:
            x = point[0]
            y = point[1]

            if math.isnan(x) or math.isnan(y):
                continue
            
            # canonical top left
            x_floor = math.floor(x)
            y_floor = math.floor(y)

            # Check the upper or lower triangle
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

            # Get a new coordinate in the warped mesh
            x_warp = x_anc + x_basis_x * x_res + y_basis_x * y_res
            y_warp = y_anc + x_basis_y * x_res + y_basis_y * y_res

            new_point_list.append((x_warp, y_warp))

        return np.array(new_point_list)

    def _con(self, ins, xlim, ylim, randx, randy, randr):
        """Trigonometric warping

        Args:
            ins (np.array): coordinates
            xlim (list): x-coordinate bounds
            ylim (list): y-coordinate bounds
            randx (_type_): Vertical distortion amplitude
            randy (_type_): Horizontal distortion amplitude
            randr (_type_): Inclination amplitude

        Returns:
            np.array: warped coordinates
        """
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
                else:
                    continue

            if i in ylim:
                new_point.append(i)
            else:
                if ylim[0] < i+offset_y < ylim[1]:
                    new_point.append(i+offset_y)
                else:
                    continue

            new_point_list.append(np.array(new_point))
        
        return np.array(new_point_list)

    def _gaussian_grid(self, def_args):
        """Get Gaussian mesh

        Args:
            def_args (dict): Parameters of the Gaussian distortion: mean and standard

        Returns:
            tulpe: The x and y coordinates of the grid
        """
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

    def add_centerline(self, aff=None):
        """Add a path: a center line w/ lanes and crosswalks in series"""
        centerlines = self.geom_dict['centerline']

        # Obtain the centerline that needs to be disturbed according to the disturbance ratio or DIY(manual selection)
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
            origin_rot = [0,0]
            xfact = 1
            yfact = 1
            origin_sca = Point([0,0])
            
            if self.tran_args.diy: # Manually set perturbation parameters
                print('adding centerline parameters setting:')
                xoff = float(input("Enter shift xoff: "))
                yoff = float(input("Enter shift yoff: "))
                angle = int(float(input("Enter rotate angle: ")))
                xfact = int(float(input("Enter scale xfact(can only be 1 or -1): ")))
                yfact = int(float(input("Enter scale yfact(can only be 1 or -1): ")))
            else:
                if aff is None:
                    xoff = random.choice([-3.5, 3.5])
                    yoff = random.choice([-3.5, 3.5])
                    angle = random.randint(-180, 180)
                    xfact = random.choice([-1, 1])
                    yfact = random.choice([-1, 1])
                else: # When add_centerline as an intermediate step of the affine_transform_centerline
                    if aff['name'] == 'shi_lan':
                        xoff = random.uniform(aff['tran'][2][0][0], aff['tran'][2][0][1])
                        yoff = random.uniform(aff['tran'][2][1][0], aff['tran'][2][1][1])
                    elif aff['name'] == 'rot_lan':
                        angle = random.randint(aff['tran'][2][0][0], aff['tran'][2][0][1])
                    elif aff['name'] == 'sca_lan':
                        xfact = self.tran_args.sca_pat[2][0]
                        yfact = self.tran_args.sca_pat[2][1]
            
            # Convert and add centerline            
            new_cl = centerline_dic['geom']
            new_cl = affine_transfer_4_add_centerline(new_cl, xoff, yoff, angle, origin_rot, xfact, yfact, origin_sca)
            new_cl_dic = layer_dict_generator(new_cl, source='new')
            self.geom_dict['centerline'][new_cl_dic['token']] = new_cl_dic
            
            # new lane from new centerline 
            center_lane = new_cl.buffer(1.8)
            new_lane_dic = layer_dict_generator(center_lane, source='centerline')
            self.geom_dict['lane'][new_lane_dic['token']] = new_lane_dic
            
            self.geom_dict['lane'][new_lane_dic['token']]['centerline_token'] = [new_cl_dic['token']] #add centerline token to new lane
            self.geom_dict['centerline'][new_cl_dic['token']]['lane_token'] = [new_lane_dic['token']] #add lane token to centerline
            
            # Convert and add lanes that connected with new centerline 
            if 'lane_token' in centerline_dic:
                for lane_token in centerline_dic['lane_token']:
                    if lane_token in self.geom_dict['lane']:
                        lane_dic = self.geom_dict['lane'][lane_token]
                        if lane_dic['from'] != 'lane_connector':
                            new_cl = lane_dic['geom']
                            new_cl = affine_transfer_4_add_centerline(new_cl, xoff, yoff, angle, origin_rot, xfact, yfact, origin_sca)
                            new_lane_dic = layer_dict_generator(new_cl, source='new')
                            self.geom_dict['lane'][new_lane_dic['token']] = new_lane_dic #new lane from lane
                            
                            self.geom_dict['lane'][new_lane_dic['token']]['centerline_token'] = [new_cl_dic['token']] #add centerline token to new lane
                            self.geom_dict['centerline'][new_cl_dic['token']]['lane_token'].append(new_lane_dic['token']) #add lane token to centerline
            
            # Convert and add ped_crossing that connected with new centerline  #FIXME
            # self.geom_dict['centerline'][centerlien_dic['token']]['ped_crossing_token'] = []        
            # if 'ped_crossing_token' in centerline_dic:
            #     for lane_token in centerline_dic['ped_crossing_token']:
            #         lane_dic = self.geom_dict['ped_crossing'][lane_token]
            #         new_cl = lane_dic['geom']
            #         new_cl = affine_transfer_4_add_centerline(new_cl, xoff, yoff, angle, origin, xfact, yfact)
            #         new_lane_dic = layer_dict_generator(new_cl, source='new')
            #         self.geom_dict['ped_crossing'][new_lane_dic['token']] = new_lane_dic
                    # self.geom_dict['ped_crossing'][new_lane_dic['token']]['centerline_token'] = [centerlien_dic['token']] #add centerline token to new ped_crossing
                    # self.geom_dict['centerline'][centerlien_dic['token']]['ped_crossing_token'].append(new_lane_dic['token']) #add lane token to centerline

    def del_centerline(self, aff=None):
        """Delet a path: a center line w/ lanes and crosswalks in series"""
        centerlines = self.geom_dict['centerline']
        road_segments = self.geom_dict['boundary']
        lanes = self.geom_dict['lane']
        lane_dividers = self.geom_dict['divider']
        ped_crossings = self.geom_dict['ped_crossing']
        agents = self.geom_dict['agent']
        
        # Obtain the centerline that needs to be disturbed according to the disturbance ratio or DIY(manual selection)
        if self.tran_args.diy:
            self.delet_centerlines = geometry_manager(self.geom_dict, 'centerline', ['lane', 'ped_crossing', 'boundary'])
        else:
            if aff is not None:
                times = math.ceil(len(centerlines.keys()) * aff['tran'][1])
            else:
                times = math.ceil(len(centerlines.keys()) * self.tran_args.del_lan[1])
            if times == 0:
                return self.geom_dict

            self.delet_centerlines = random_select_element(centerlines, times)

        # check if there are map layers in use after removing centerlines
        # remove a centerline need also to remove its token from connected lanes and ped_crossings
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

        # check if there are agents in use or need an update after removing a lane
        # remove a lane need also to remove connected agent and agent trajectory
        # If an agent exits but is trajectory longer being used, update it
        del_agents = []    
        for del_lane in delet_lanes:
            if 'agent_eco_token' in del_lane.keys():
                for agent_token in del_lane['agent_eco_token']:
                    del_lane['agent_token'].remove(agent_token)
                    del_agents.append(agent_token)
                    
            if 'agent_token' in del_lane.keys():    
                for ag_tra_token in del_lane['agent_token']:
                    if ag_tra_token in agents.keys():
                        agents[ag_tra_token]['lane_tokens'] = []
        
        for ag_token, ag_val in agents.items():
            if not len(ag_val['lane_token']):
                if len(ag_val['eco_lane_token']):
                    centerline_candidates = lanes[ag_val['eco_lane_token'][0]]['centerline_token']
                    if len(centerline_candidates):
                        new_tra_token = random.choice(centerline_candidates)
                        ag_val['lane_token'] = centerlines[new_tra_token]['lane_tokens']
                    else:
                        del_agents.append(ag_token)

        for ag_token in set(del_agents):
            if 'lane_token' in agents[ag_token]:
                for l_token in agents[ag_token]['lane_token']:
                    if l_token in lanes:
                        try:
                            lanes[lane_token][agent_token].remove(ag_token)
                        except:
                            continue
                        
            agents.pop(ag_token)
        
        # check if there are dividers in use
        delet_dividers = []
        for delet_lane in delet_lanes:
            for divider_token in ['left_lane_divider_token', 'right_lane_divider_token']:
                if divider_token in delet_lane:
                    delet_dividers.append(delet_lane[divider_token])

            # check if there are overlap with road_segment
            for road_dic in road_segments.values():
                new_road_seg = keep_non_intersecting_parts(
                    road_dic['geom'], delet_lane['geom'])
                if len(new_road_seg) == 1:
                    road_dic['geom'] = new_road_seg[0]
                elif len(new_road_seg) > 1:
                    road_dic['geom'] = MultiPolygon(new_road_seg)
                else:
                    continue

        for token in delet_dividers:
            if token in lane_dividers:
                lane_dividers.pop(token)

        # check unreasonable: remove isolated element
        road_segments = check_isolated_new(road_segments, [centerlines])
        lane_dividers = check_isolated_new(lane_dividers, [road_segments, lanes])
        ped_crossings = check_isolated_new(ped_crossings, [road_segments, lanes], True)

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

    def adjust_lane_width(self):
        """Widen or narrow the boundaries of a path: the boundaries of all lanes and crosswalks connected by the center line"""
        move_distance = self.tran_args.wid_lan[2] / 2.0
        
        # Obtain the lane(centerline) that needs to be disturbed according to the disturbance ratio or DIY(manual selection)
        if self.tran_args.diy:
            # select_centerlines = geometry_manager(self.geom_dict, 'centerline', ['lane', 'ped_crossing', 'boundary'])
            pass #TODO
        else:
            times = math.ceil(len(self.geom_dict_croped['centerline']) * self.tran_args.wid_lan[1])
            if times == 0:
                return self.geom_dict_croped

            centerline_dict = {}
            for ind, cl in enumerate(self.geom_dict_croped['centerline']):
                cl_dic = {}
                cl_dic['token'] = str(ind)
                cl_dic['geom'] = cl
                centerline_dict[str(ind)] = cl_dic
                
            select_centerlines = random_select_element(centerline_dict, times)
            
        for cl_dic in select_centerlines:
            # Get the centerline and its center as the starting point for the move
            centerline = cl_dic['geom']
            centerline_center = Point(centerline.centroid)
            
            for layer, polylines in self.geom_dict_croped.items():
                moved_polylines = []
                if layer in ['ped_crossing', 'lane']: # Move disjointly ped_crossing and lane
                    for polyline in polylines:
                        if not centerline.intersection(polyline):
                            moved_polylines.append(move_geom(centerline_center, polyline, move_distance))
                        else:
                            moved_polylines.append(polyline)
                    self.geom_dict_croped[layer] = moved_polylines
                    
                elif layer == 'centerline': # Move non-own disjoint centerline
                    for cl_k, polyline in centerline_dict.items():
                        if cl_k == cl_dic['token']:
                            continue
                        else:
                            if polyline['geom'].intersection(centerline):
                                continue
                            else:
                                polyline['geom'] = move_geom(centerline_center, polyline['geom'], move_distance)
                        
                else: # Move other map layers
                    for polyline in polylines:
                        moved_polylines.append(move_geom(centerline_center, polyline, move_distance))
                    self.geom_dict_croped[layer] = moved_polylines

        self.geom_dict_croped['centerline'] = [geom['geom'] for geom in centerline_dict.values()]
    
    def adjust_boundary(self):
        """Widen or narrow the boundaries"""
        move_distance = self.tran_args.wid_bou[2] / 2.0
        
        # Obtain the boundary that needs to be disturbed according to the disturbance ratio or DIY(manual selection)
        if self.tran_args.diy:
            # select_boundaris = geometry_manager(self.geom_dict_croped, 'boundary', ['ped_crossing', 'divider'])
            pass #TODO
        else:
            times = math.ceil(len(self.geom_dict_croped['boundary']) * self.tran_args.wid_bou[1])
            if times == 0:
                return self.geom_dict_croped
            
            boundary_dict = {}
            for ind, bd in enumerate(self.geom_dict_croped['boundary']):
                bd_dic = {}
                bd_dic['token'] = str(ind)
                bd_dic['geom'] = bd
                boundary_dict[str(ind)] = bd_dic
                
            select_boundaries = random_select_element(boundary_dict, times)
        
        # Move the boundary line around the map center [0,0]
        for bd_dic in select_boundaries:
            boundary = bd_dic['geom']
            center = Point([0,0])
            
            boundary_dict[bd_dic['token']]['geom'] = move_geom(center, boundary, move_distance)
                            
        self.geom_dict_croped['boundary'] = [geom['geom'] for geom in boundary_dict.values()]

    def add_ped_crossing(self):
        """Add a ped_crossing"""

        # Select the road block to create the crosswalk according to the disturbance ratio or DIY(manual selection)
        if self.tran_args.diy:
            selected_road_seg = geometry_manager(self.geom_dict, 'boundary', ['ped_crossing', 'divider'])
        else:
            times = math.ceil(len(self.geom_dict['boundary']) * self.tran_args.add_ped[1])
            if times == 0:
                return self.geom_dict

            selected_road_seg = random_select_element(self.geom_dict['boundary'], times)
            
        for road_seg in selected_road_seg:
            # Create a new crosswalk
            new_ped = self._creat_ped_polygon(road_seg['geom'])
            new_ped_dic = layer_dict_generator(new_ped, source='new')
            self.geom_dict['ped_crossing'][new_ped_dic['token']] = new_ped_dic
            
            # link the new crosswalk to exit centerlines in the map
            self.geom_dict['ped_crossing'][new_ped_dic['token']]['centerline_token'] = []
            for cl in self.geom_dict['centerline']:
                if cl['geom'].intersection(new_ped):
                    self.geom_dict['ped_crossing'][new_ped_dic['token']]['centerline_token'].append(cl['token'])
                    if 'ped_crossing_token' not in cl:
                        cl['ped_crossing_token'] = []
                    cl['ped_crossing_token'].append(new_ped_dic['token'])
            
    def delete_layers(self, layer_name, args):
        """delete map layer

        Args:
            layer_name (str): name of map layer
            args (dict): the parameters of deleting layer
        """
        times = math.ceil(len(self.geom_dict[layer_name]) * args[1])
        if times:
            delet_ped = random_select_element(self.geom_dict[layer_name], times)
            for ped in delet_ped:
                self.geom_dict[layer_name].pop(ped['token'])

    def shift_ped_crossing(self):
        """Move the crosswalk along the center line"""
        
        # Select the crosswalk according to the disturbance ratio or DIY(manual selection)
        if self.tran_args.diy:
            select_peds = geometry_manager(self.geom_dict, 'ped_crossing', ['lane', 'centerline', 'boundary'])
        else:
            times = math.ceil(len(self.geom_dict['ped_crossing'].keys()) * self.tran_args.shi_ped[1])
            if times == 0:
                return self.geom_dict

            select_peds = random_select_element(self.geom_dict['ped_crossing'], times)
            
        for ped_cro in select_peds:
            # Select a centerline that passes through the crosswalk
            if 'centerline_token' in ped_cro:
                if len(ped_cro['centerline_token']):
                    along_cl_token = random.choice(ped_cro['centerline_token'])
                    along_cl = self.geom_dict['centerline'][along_cl_token]
                    if 'lane_token' in along_cl:
                        if len(along_cl['lane_token']):
                            # Move the crosswalk forward and backward along the center line
                            moved_ped_forward, moved_ped_backward = move_polygon_along_polyline(ped_cro['geom'], along_cl['geom'], self.tran_args.shi_ped[2])
                            # Verify that the moved crosswalk is still on the lane
                            for moved_ped in moved_ped_forward, moved_ped_backward:
                                check = 0
                                for l_token in along_cl['lane_token']:
                                    lane = self.geom_dict['lane'][l_token]
                                    if moved_ped.intersection(lane['geom']):
                                        check = 1
                                        break
                                
                                if check:
                                    self.geom_dict['ped_crossing'][ped_cro['token']]['geom'] = moved_ped
                                    self.geom_dict['ped_crossing'][ped_cro['token']]['from'] = 'new'
                                    break

    def affine_transform_patch(self):
        """affine transform for all the map geometry elements"""
        
        # Affine transformation parameter settings
        if self.tran_args.diy:
            rot_p = input("Enter rotate angle: ")
            rot_p = [int(float(rot_p)), [0,0]]
            sca_p_x = input("Enter scale xfact(can only be 1 or -1): ")
            sca_p_y = input("Enter scale yfact(can only be 1 or -1): ")
            sca_p = [int(float(sca_p_x)), int(float(sca_p_y)), [0,0]]
            shi_p_x = input("Enter shift xoff: ")
            shi_p_y = input("Enter shift yoff: ")
            shi_p = [float(shi_p_x), float(shi_p_y)]
        else:
            if self.tran_args.rot_pat[0]:
                rot_p = [random.randint(self.tran_args.rot_pat[2][0][0], self.tran_args.rot_pat[2][0][1]), self.tran_args.rot_pat[2][1]]
            if self.tran_args.sca_pat[0]:
                sca_p = [self.tran_args.rot_pat[2][0], self.tran_args.rot_pat[2][1], self.tran_args.sca_pat[2][2]]
            if self.tran_args.shi_pat[0]:
                shi_p = [random.uniform(self.tran_args.shi_pat[2][0][0], self.tran_args.shi_pat[2][0][1]),
                        random.uniform(self.tran_args.shi_pat[2][1][0], self.tran_args.shi_pat[2][1][1])]
        
        # Perform an affine transformation on all map geometry layers
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

    def shift_map(self):
        """shift vector map"""
        
        # shift parameter settings
        xoff = random.uniform(self.tran_args.shi_pat[2][0][0], self.tran_args.shi_pat[2][0][1])
        yoff = random.uniform(self.tran_args.shi_pat[2][1][0], self.tran_args.shi_pat[2][1][1])
        
        # shift each coordinate point of the vector map
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
        """rotate vector map"""
        
        # rotate angle parameter settings
        angle = random.randint(self.tran_args.rot_pat[2][0][0], self.tran_args.rot_pat[2][0][1])
        
        # rotate each coordinate point of the vector map
        for layer, vect_list in self.vect_dict.items():
            if len(vect_list):
                for ind, vect in enumerate(vect_list):
                    new_point_list = []
                    for point in vect:
                        x = point[0] * math.cos(angle) - point[1] * math.sin(angle)
                        y = point[0] * math.sin(angle) + point[1] * math.cos(angle)
                        
                        new_point_list.append([x, y])

                    self.vect_dict[layer][ind] = np.array(new_point_list)
    
    def flip_map(self):
        """flip vector map by scaling with xfact ane yfact only be 1 or -1"""
        
        # scale each coordinate point of the vector map
        for layer, vect_list in self.vect_dict.items():
            if len(vect_list):
                for ind, vect in enumerate(vect_list):
                    new_point_list = []
                    for point in vect:
                        x = point[0] * self.tran_args.sca_pat[2][0]
                        y = point[1] * self.tran_args.sca_pat[2][1]
                        new_point_list.append([x, y])

                    self.vect_dict[layer][ind] = np.array(new_point_list)

    def difromate_map(self):
        """Trigonometric warping vector map"""
        
        ## trigonometric warping parameter settings
        v =0 # Vertical distortion amplitude random maximum range int
        h =0 # Horizontal distortion amplitude random maximum range int
        i =0 # Inclination amplitude [-Max_r Max_r] int
        
        if self.tran_args.diy:
            i = float(input("Enter your inclination amplitude: "))
        else:
            v = self.tran_args.def_pat_tri[2][0]
            h = self.tran_args.def_pat_tri[2][1]
            i = self.tran_args.def_pat_tri[2][2]  

        # xy coordinate area
        xlim = [-self.patch_box[3]/2, self.patch_box[3]/2]
        ylim = [-self.patch_box[2]/2, self.patch_box[2]/2]

        # Trigonometric warping all vectors on map
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
   
    def guassian_warping(self):
        """Gaussian warping map patches"""
        
        # mean and standard settings
        if self.tran_args.diy:
            self.tran_args.def_pat_gau[2][0] = float(input("Enter gaussian mean: "))
            self.tran_args.def_pat_gau[2][1] = float(input("Enter gaussian standard: "))
        
        # get gaussian warping mesh
        g_xv, g_yv = self._gaussian_grid(self.tran_args.def_pat_gau)

        # apply mesh to all vectors in map
        for key in self.vect_dict.keys():
            if len(self.vect_dict[key]):
                new_list = []
                for ins in self.vect_dict[key]:
                    if ins.size:
                        ins = fix_corner(ins, self.patch_box)
                        new_list.append(self._warping(ins, g_xv, g_yv))
                
                self.vect_dict[key] = new_list

    def guassian_noise(self):
        """add Gaussian noise to map patches"""
        
        # mean and standard settings
        if self.tran_args.diy:
            self.tran_args.noi_pat_gau[2][0] = input("Enter gaussian mean: ")
            self.tran_args.noi_pat_gau[2][1] = float(input("Enter gaussian standard: "))
        
        # add gaussian noise mesh to vector map
        for key in self.vect_dict.keys():
            if len(self.vect_dict[key]):
                for ind, ins in enumerate(self.vect_dict[key]):
                    g_nois = np.random.normal(
                        self.tran_args.noi_pat_gau[2][0], self.tran_args.noi_pat_gau[2][1], ins.shape)
                    self.vect_dict[key][ind] += g_nois

    def truncate_and_save(self, map_type, map_name = 'inter_pt'):
        """ Visualizing the intermediate steps of perturbations

        Args:
            map_type (str): geometry or vectory
            map_name (str, optional): output name. Defaults to 'inter_pt'.
        """
        
        if self.tran_args.truncate:
            if map_type == 'geom':
                ## transfer geom to ready-show vectorized geom 
                trans_dic = self.vector_map.gen_vectorized_samples(self.geom_dict)
                ## Transform vectorized geom to NumPy array
                map_vect_dic = geom_to_np(trans_dic)
            elif map_type == 'geom_ins':
                map_vect_dic = geom_to_np(self.geom_dict)
            elif map_type == 'vect':
                map_vect_dic = self.vect_dict
                
            self.visual.vis_contours(map_vect_dic, self.ann_name + '_truncate_' + map_name)
            
    def perturb_vect_map(self):
        """Perturb the map vector layer: image algorithms acting on numpy arrays"""
        
        # tranform the map geometry layer to vector
        self.vect_dict = geom_to_np(self.geom_dict_croped, inter=True)

        if self.tran_args.shi_pat[0]:
            self.shift_map()
            self.truncate_and_save('vect', '10_shift_map')
            
        if self.tran_args.rot_pat[0]:
            self.rotate_map()
            self.truncate_and_save('vect', '11_rotate_map')
            
        if self.tran_args.sca_pat[0]:
            self.flip_map()
            self.truncate_and_save('vect', '12_flip_map')
        
        if self.tran_args.def_pat_tri[0]:
            self.difromate_map()
            self.truncate_and_save('vect', '13_warping_tri')

        if self.tran_args.def_pat_gau[0]:
            self.guassian_warping()
            self.truncate_and_save('vect', '14_warping_gua')

        if self.tran_args.noi_pat_gau[0]:
            self.guassian_noise()
            self.truncate_and_save('vect', '15_noiseing')

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
        
        if self.tran_args.del_ped[0]:
            self.delete_layers('ped_crossing', self.tran_args.del_ped)
            self.truncate_and_save('geom', '4_delet_ped_crossing')
        
        if self.tran_args.add_ped[0]:
            self.add_ped_crossing()
            self.truncate_and_save('geom', '5_add_ped_crossing')
        
        if self.tran_args.shi_ped[0]:
            self.shift_ped_crossing()
            self.truncate_and_save('geom', '6_shift_ped_crossing')
        
        if self.tran_args.del_div[0]:
            self.delete_layers('divider', self.tran_args.del_div)
            self.truncate_and_save('geom', '7_delet_divider')
            
        # if any(getattr(self.tran_args, pat)[0] for pat in ['rot_pat', 'sca_pat', 'shi_pat']):
        #     self.affine_transform_patch()
        #     self.truncate_and_save('geom', '7_affine_transform_map')
        
        self.geom_dict_for_json = copy.deepcopy(self.geom_dict)
        #Converting an overlay geometry layer to a geometry instance
        self.geom_dict = self.vector_map.gen_vectorized_samples(self.geom_dict)
        
        # crop map with a big patch box to get LineString instance from a big polygon map layer
        self.geom_dict_croped = {}
        for vec_class, pt_geom_dic in self.geom_dict.items():
            self.geom_dict_croped[vec_class] = get_geom_in_patch(self.map_explorer, pt_geom_dic, patch_box=[0,0,max(self.patch_box[2:]), max(self.patch_box[2:])])
        
        if self.tran_args.wid_lan[0]:
            self.adjust_lane_width() #FIXME
            self.truncate_and_save('geom_ins', '8_adjust_lane')
            
        if self.tran_args.wid_bou[0]:
            self.adjust_boundary()
            self.truncate_and_save('geom_ins', '9_adjust_boundary')

from mapmodex import MapModEX

## general setting
data_root = '/home/li/Documents/map/data/sets'
output_path = '/home/li/Documents/map/MapModEX/output'
map_version_nusc = ['v1.0-mini']
# map_version_av2 = ['test']

## load MapModEx
mme = MapModEX(data_root, output_path)

## set perturbation versions
pt_geom_1 = {'del_lan':[1, 0.3, None], 'pt_name':'delete_lane'}   # delet = delet
pt_geom_2 = {'add_lan':[1, 0.1, None], 'pt_name':'add_lane'}   # add = copy + shift + rotate + flip + past
pt_geom_3 = {'rot_pat':[1, None, [[-30,30], [0,0]]], 'shi_pat':[1, None, [[-1.5, 1.5], [-1.5, 1.5]]], 'pt_name':'affine_transform_map'}  # affine transformation                  : widen / narrow
pt_vect_1 = {'def_pat_gau':[1, None, [0, 1]], 'pt_name':'warping_map'}  # warping, with mean=0 and standard=1
pt_vect_2 = {'noi_pat_gau':[1, None, [0, 0.25]], 'pt_name':'noising_map'}  # noise
pt_multi = {'del_lan':[1, 0.3, None], 'add_lan':[1, 0.1, None], 'rot_pat':[1, None, [[-30,30], [0,0]]], 'shi_pat':[1, None, [[-1.5, 1.5], [-1.5, 1.5]]], 
            'def_pat_gau':[1, None, [0, 1]], 'noi_pat_gau':[1, None, [0, 1]], 'pt_name':'multi_pertubation'}   # all
pt_del_div = {'del_lan':[1, 0, None], 'pt_name':'delete_lane_div', 'diy':True}
pt_add_div = {'add_lan':[1, 0.1, None], 'pt_name':'add_lane_div', 'diy':True}

pt_versions = [pt_multi]
mme.update_pt_version(pt_versions, False)

## lunch MapModEx
mme.mod_nuscenes(map_version_nusc, output_type='json', vis=True)
# mme.mod_av2(map_version_av2, output_type='pkl', vis=True)

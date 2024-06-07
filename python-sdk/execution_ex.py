from mapmodex import MapModEX

## general setting
data_root = './data'
output_path = './ouput' # optional. default is the current path.

map_version_nusc = ['v1.0-mini'] #'v1.0-trainval', 'v1.0-test'
# map_version_av2 = ['test'] #'train', 'val'
# mv_mme = ['mme_org'] #The file name used to save the MapModEX map is usually the perturbed version name.

## load MapModEx
mme = MapModEX(data_root, output_path)

## set perturbation versions
pt_geom_1 = {'del_lan':[1, 0.3, None], 'pt_name':'delete_lane'}

pt_vect_1 = {'def_pat_tri':[1, None, [0,0,1.5]], 'pt_name':'warping_map'}  #Warping: horizontal, vertical, and inclination distortion = 0, 0, 1.5

pt_multi = {'add_lan':[1, 0.1, None], 'wid_lan':[1, 0.1, 2], 'rot_pat':[1, None, [[-30,30], [0,0]]], 'def_pat_tri':[1, None, [2,2,2]],
            'noi_pat_gau':[1, None, [0, 0.05]], 'pt_name':'multi_pertubation'}

pt_del_div = {'del_lan':[1, 0, None], 'pt_name':'delete_lane_div', 'diy':True} #In DIY mode, customize layers and parameters.

pt_versions = [pt_geom_1]
mme.update_pt_version(pt_versions, False)

## launch MapModEx
mme.mod_nuscenes(map_version_nusc, output_type='json', vis=True) #output_type=None gives no 'json' or 'pkl' file. 'vis' controls visualization.
# mme.mod_av2(map_version_av2, output_type='pkl', vis=True)
# mme.mod_mme(mv_mme, output_type='pkl')

from mapmodex import MapModEX

## general setting
data_root = './data'
output_path = './ouput' # optional. default is the current path.

map_version_nusc = ['v1.0-mini'] #'v1.0-trainval', 'v1.0-test'
# map_version_av2 = ['test'] #'train', 'val'
# mv_mme = ['mme_org'] #The file name to save the map generated from MapModEX to, usually the name of the perturbed version. 

## load MapModEx
mme = MapModEX(data_root, output_path)

## set perturbation versions
pt_geom_1 = {'del_lan':[1, 0.3, None], 'pt_name':'delete_lane'}   # delet = delet

pt_vect_1 = {'def_pat_tri':[1, None, [0,0,1.5]], 'pt_name':'warping_map'}  # warping, with mean=0 and standard=1

pt_multi = {'add_lan':[1, 0.1, None], 'wid_lan':[1, 0.1, 2], 'rot_pat':[1, None, [[-30,30], [0,0]]], 'def_pat_tri':[1, None, [2,2,2]],
            'noi_pat_gau':[1, None, [0, 0.05]], 'pt_name':'multi_pertubation'}

pt_del_div = {'del_lan':[1, 0, None], 'pt_name':'delete_lane_div', 'diy':True}

pt_versions = [pt_geom_1]
mme.update_pt_version(pt_versions, False)

## launch MapModEx
mme.mod_nuscenes(map_version_nusc, output_type='json', vis=True)
# mme.mod_av2(map_version_av2, output_type='pkl', vis=True)
# mme.mod_mme(mv_mme, output_type='pkl')

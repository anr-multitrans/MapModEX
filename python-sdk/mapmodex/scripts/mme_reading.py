import os
import mmcv

from nuscenes.map_expansion.map_api import NuScenesMapExplorer

from .map_reading import get_vec_map
from ..utils import *


def creat_mme_infos(root_path, map_version, pertube_vers, out_path, pc_range=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0], vis=False,
                    output_type='json'):
    """Create info file of mme dataset.

    Args:
        root_path (str): Path of the data root.
        map_version (list): Version of the data.
        pertube_vers (list): version of perturbation.
        out_path (_type_): output path.
        pc_range (list, optional): patch box size(3D). Defaults to [-30.0, -15.0, -5.0, 30.0, 15.0, 3.0].
        vis (bool, optional): visualization. Defaults to False.
        output_type (str, optional): output type. 'pkl' is the data used for model training, and 'json' is the map data. Defaults to 'json'.
    """
    mme_infos = []
    for scene in mmcv.track_iter_progress(os.listdir(root_path)):
        scene_path = os.path.join(root_path, scene)
        for simple in mmcv.track_iter_progress(os.listdir(scene_path)):
            info = {}
            info['scene_token'] = scene
            info['token'] = simple
            for v in map_version:
                temp_info = {}
                temp_info['scene_token'] = scene
                temp_info['token'] = simple
                temp_info['dataset'] = 'mme'
                pt_dataroot = os.path.join(scene_path, simple, v)
                ns_map = NuScenesMap4MME(pt_dataroot)
                ns_mapex = NuScenesMapExplorer(ns_map)
                gma = get_vec_map(temp_info, ns_map, ns_mapex, out_path, pc_range=pc_range, out_type=output_type, vis=vis, mme=True)
                info[v] = gma.get_map_ann(pertube_vers)

            mme_infos.append(info)
            
    if output_type == 'pkl':
        info_path = os.path.join(out_path, 'mme_map_infos.pkl')
        mmcv.dump(mme_infos, info_path)

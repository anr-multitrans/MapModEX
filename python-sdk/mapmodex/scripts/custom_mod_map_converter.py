import json
import os

from matplotlib import pyplot as plt
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer

from .data_reading import VectorizedLocalMap
from ..utils import *


colors_plt = {'divider': '#808000', 'ped_crossing': '#000080', 'boundary': '#008000', 'centerline': 'y'}


def vis_contours(contours, patch_box, save_path=None, show=False):
    plt.figure(figsize=(2, 4))
    plt.xlim(-patch_box[3]/2, patch_box[3]/2)
    plt.ylim(-patch_box[2]/2, patch_box[2]/2)
    plt.axis('off')
    
    map_classes=['ped_crossing', 'boundary', 'divider']
    for pred_label_3d in map_classes:
        if len(contours[pred_label_3d]):
            for pred_pts_3d in contours[pred_label_3d]:
                pts_x = pred_pts_3d[:, 0]
                pts_y = pred_pts_3d[:, 1]
                plt.plot(
                    pts_x, pts_y, color=colors_plt[pred_label_3d], linewidth=1, alpha=0.8, zorder=-1)
                plt.scatter(
                    pts_x, pts_y, color=colors_plt[pred_label_3d], s=1, alpha=0.8, zorder=-1)

    if save_path is not None:
        check_path(save_path)
        map_path = os.path.join(save_path, 'singapore-onenorth.png')
        plt.savefig(map_path, bbox_inches='tight',
                    format='png', dpi=1200)
        print("saved pic to:", save_path)

    if show:
        plt.show()

    plt.close()


def vis_json(json_fpath, json_fname):
    with open(os.path.join(json_fpath, json_fname), 'r') as fh:
        json_obj = json.load(fh)
    
    patch_box = [0, 0, 62, 32]
    vis_fpath = os.path.join(json_fpath, os.path.splitext(json_fname)[0]+".png")
    
    vis_contours(json_obj, patch_box, vis_fpath)


class NuScenesMap4MME(NuScenesMap):
    def __init__(self,
                 dataroot: str = '/data/sets/nuscenes',
                 map_name: str = 'singapore-onenorth'):
        
        super(NuScenesMap4MME, self).__init__(dataroot, map_name)
        
        self.boundary = self._load_layer('boundary')
        self.divider = self._load_layer('divider')
        
        # super(NuScenesMap4Mod, self).__init__(dataroot, map_name)

        
if __name__ == '__main__':
    dataroot = "/home/li/Documents/map/MapTRV2Local/tools/maptrv2/map_perturbation/pt_map"
    scene = "2fc3753772e241f2ab2cd16a784cc680"
    simple = "1b90013f0287408f9371729424c93892"
    pt_version = "mme_1"
    pt_dataroot = os.path.join(dataroot, scene, simple, pt_version)
    
    json_fname = "singapore-onenorth"

    patch_size = (60, 30)
    
    # ns_map = NuScenesMap(pt_dataroot, json_fname)
    ns_map = NuScenesMap4MME(pt_dataroot, json_fname)
    ns_mapex = NuScenesMapExplorer(ns_map)
    vector_map = VectorizedLocalMap(ns_map, ns_mapex, patch_size)
    
    trans_dic = vector_map.gen_vectorized_samples_by_pt_json()

    #visualization
    # vis_fpath = os.path.join(pt_dataroot, os.path.splitext(json_fname)[0]+".png")
    trans_np_dict_4_vis = vector_map.geom_to_np(trans_dic['map_ins_dict'], 20)
    vis_contours(trans_np_dict_4_vis, trans_dic['patch_box'], pt_dataroot)

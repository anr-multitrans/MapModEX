###     Based on data processing code from the official MapTRv2 code available under MIT License
###     Original code can be found at https://github.com/hustvl/MapTR/blob/maptrv2/tools/maptrv2

import os
import mmcv

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils import splits
from nuscenes.eval.common.utils import Quaternion

from .map_reading import get_vec_map


class MMENuScenesMapExplorer(NuScenesMapExplorer):
    """Class inheriting from NuScenesMapExplorer that adds information about lane dividers to their corresponding lanes.

    Args:
        NuScenesMapExplorer (NuScenesMapExplorer): Dataset class in the nuScenes map dataset explorer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._link_divider_to_lane()
    
    def _link_divider_to_lane(self):
        """Adds information about lane dividers to their corresponding lanes."""
        for lane in self.map_api.lane:
            for l_r in ['left_lane_divider', 'right_lane_divider']:
                if len(lane[l_r+'_segments']):
                    lane_segments = set([seg['node_token'] for seg in lane[l_r+'_segments']])
                    lane_segment_nodes = set([seg['token'] for seg in lane[l_r+'_segment_nodes']])
                    for divider in self.map_api.lane_divider:
                        if len(divider['lane_divider_segments']):
                            divider_segments = set([seg['node_token'] for seg in divider['lane_divider_segments']])
                            line_nodes = set(divider['node_tokens'])
                            if self.has_equal_list([lane_segments, lane_segment_nodes], [divider_segments, line_nodes]):
                                if 'lane_token' not in divider:
                                    divider['lane_token'] = []
                                divider['lane_token'].append(lane['token'])
                                lane[l_r+'_token'] = divider['token']

    def has_equal_list(self, team1, team2):
        """
        Determine whether at least one list in each team contains the same elements in any order.
        
        :param team1: A list of two lists representing the first team.
        :param team2: A list of two lists representing the second team.
        :return: True if at least one list in each team contains the same elements, False otherwise.
        """
        # Iterate through each list in team1
        for list1 in team1:
            # Iterate through each list in team2
            for list2 in team2:
                # Check if the lists contain the same elements in any order
                if list1 == list2:
                    return True
        
        return False

def _get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of essential information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is the absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

def _obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep

def _obtain_vectormap(nusc, nusc_maps, map_explorer, info, point_cloud_range, pertube_vers, out_type, out_path, vis=False):
    """Get Vector Map

    Args:
        nusc (NuScenes): Dataset class in the nuScenes dataset.
        nusc_maps (NuScenesMap): Dataset class in the nuScenes map dataset.
        map_explorer (NuScenesMapExplorer): Dataset class in the nuScenes map dataset explorer.
        info (dict): information from the original map
        point_cloud_range (list): patch box size(3D).
        pertube_vers (list): perturbed versions, each version should be a dict with parameter_names and parameter_values.
        out_type (str): output type. 'pkl' is the data used for model training, and 'json' is the map data.
        out_path (str): output path.
        vis (bool, optional): visulization. Defaults to False.

    Returns:
        dict: information include vector map layers
    """
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = info['lidar2ego_translation']
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(
        info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = info['ego2global_translation']

    lidar2global = ego2global @ lidar2ego

    lidar2global_translation = list(lidar2global[:3, 3])
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

    location = info['map_location']

    info['dataset'] = 'nuscenes'
    gma = get_vec_map(info, nusc_maps[location], map_explorer[location], out_path, lidar2global_translation, lidar2global_rotation,
                      point_cloud_range, out_type, nusc=nusc, vis=vis)
    info = gma.get_map_ann(pertube_vers)

    return info

def _fill_trainval_infos(nusc,
                         nusc_maps,
                         map_explorer,
                         train_scenes,
                         pertube_vers,
                         out_path,
                         out_type,
                         max_sweeps=10,
                         point_cloud_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
                         vis=False):
    """ Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether to use the test mode. In the test mode, no annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of the training set and validation set that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    for i, sample in enumerate(mmcv.track_iter_progress(nusc.sample)):
        map_location = nusc.get('log', nusc.get(
            'scene', sample['scene_token'])['log_token'])['location']

        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        info = {
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'map_location': map_location,
            'scene_token': sample['scene_token'],  # temporal related info
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = _obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = _obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        info = _obtain_vectormap(
            nusc, nusc_maps, map_explorer, info, point_cloud_range, pertube_vers, out_type, out_path, vis=vis)

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos

def create_nuscenes_infos(root_path,
                          out_path,
                          out_type='json',
                          pc_range=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
                          info_prefix='nuscenes',
                          pertube_vers=[],
                          version='v1.0-trainval',
                          max_sweeps=10,
                          vis=False):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    MAPS = ['boston-seaport', 'singapore-hollandvillage',
            'singapore-onenorth', 'singapore-queenstown']
    nusc_maps = {}
    map_explorer = {}
    for loc in MAPS:
        nusc_maps[loc] = NuScenesMap(dataroot=root_path, map_name=loc)
        map_explorer[loc] = MMENuScenesMapExplorer(nusc_maps[loc])

    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_maps, map_explorer, train_scenes, pertube_vers, out_path, out_type, max_sweeps=max_sweeps, vis=vis, point_cloud_range=pc_range)

    if out_type == 'pkl':
        metadata = dict(version=version)
        if test:
            print('test sample: {}'.format(len(train_nusc_infos)))
            data = dict(infos=train_nusc_infos, metadata=metadata)
            info_path = os.path.join(out_path,
                                '{}_map_infos_temporal_test.pkl'.format(info_prefix))
            mmcv.dump(data, info_path)
        else:
            print('train sample: {}, val sample: {}'.format(
                len(train_nusc_infos), len(val_nusc_infos)))
            data = dict(infos=train_nusc_infos, metadata=metadata)
            info_path = os.path.join(out_path,
                                '{}_map_infos_temporal_train.pkl'.format(info_prefix))
            mmcv.dump(data, info_path)
            data['infos'] = val_nusc_infos
            info_val_path = os.path.join(out_path,
                                    '{}_map_infos_temporal_val.pkl'.format(info_prefix))
            mmcv.dump(data, info_val_path)

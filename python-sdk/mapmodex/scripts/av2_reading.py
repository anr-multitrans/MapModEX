###     Based on data processing code from the official MapTRv2 code available under MIT License
###     Original code can be found at https://github.com/hustvl/MapTR/blob/maptrv2/tools/maptrv2

import multiprocessing
import time
import mmcv
import logging
from pathlib import Path
from os import path as osp
from .map_reading import get_vec_map
import warnings
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer

from ..utils import *

warnings.filterwarnings("ignore")


CAM_NAMES = ['ring_front_center', 'ring_front_right', 'ring_front_left',
             'ring_rear_right', 'ring_rear_left', 'ring_side_right', 'ring_side_left']
# some fail logs as stated in av2
# https://github.com/argoverse/av2-api/blob/05b7b661b7373adb5115cf13378d344d2ee43906/src/av2/map/README.md#training-online-map-inference-models
FAIL_LOGS = [
    # official
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    # observed
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d'
]

def create_av2_infos(root_path,
                    pertube_vers,
                    dest_path,
                    info_prefix='av2',
                    split='train',
                    num_multithread=64,
                    output_type='json',
                    pc_range=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
                    vis=False):
    """Create info file of av2 dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        dest_path (str): Path to store generated file, default to root_path
        split (str): Split of the data.
            Default: 'train'
    """
    root_path_v = osp.join(root_path, split)
    
    from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
    loader = AV2SensorDataLoader(Path(root_path_v), Path(root_path_v))
    log_ids = list(loader.get_log_ids())
    for l in FAIL_LOGS:
        if l in log_ids:
            log_ids.remove(l)

    print('collecting samples...')
    start_time = time.time()
    print('num cpu:', multiprocessing.cpu_count())
    print(f'using {num_multithread} threads')

    # to suppress logging from av2.utils.synchronization_database
    sdb_logger = logging.getLogger('av2.utils.synchronization_database')
    prev_level = sdb_logger.level
    sdb_logger.setLevel(logging.CRITICAL)

    global ns_map, ns_map_exp
    ns_map = NuScenesMap(os.path.join(os.path.dirname(root_path), 'nuscenes'))
    ns_map_exp = NuScenesMapExplorer(ns_map)
    
    results = []
    for log_id in mmcv.track_iter_progress(log_ids):
        result = _get_data_from_logid(
            log_id, dest_path, loader=loader, data_root=root_path_v, pertube_vers=pertube_vers, pc_range=pc_range, output_type=output_type,vis=vis)
        results.append(result)

    if output_type == 'pkl':
        samples = []
        discarded = 0
        sample_idx = 0
        for _samples, _discarded in results:
            for i in range(len(_samples)):
                _samples[i]['sample_idx'] = sample_idx
                sample_idx += 1
            samples += _samples
            discarded += _discarded

        sdb_logger.setLevel(prev_level)
        print(f'{len(samples)} available samples, {discarded} samples discarded')

        print('collected in {}s'.format(time.time()-start_time))
        infos = dict(samples=samples)

        info_path = osp.join(dest_path,
                            '{}_map_infos_{}.pkl'.format(info_prefix, split))
        print(f'saving results to {info_path}')
        mmcv.dump(infos, info_path)

def _get_data_from_logid(log_id,
                         dest_path,
                        loader,
                        data_root,
                        pertube_vers,
                        output_type='json',
                        pc_range=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
                        vis=False):
    samples = []
    discarded = 0

    log_map_dirpath = Path(osp.join(data_root, log_id, "map"))
    vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
    if not len(vector_data_fnames) == 1:
        raise RuntimeError(
            f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
    vector_data_fname = vector_data_fnames[0]
    vector_data_json_path = vector_data_fname

    from av2.map.map_api import ArgoverseStaticMap
    avm = ArgoverseStaticMap.from_json(vector_data_json_path)
    # We use lidar timestamps to query all sensors.
    # The frequency is 10Hz
    cam_timestamps = loader._sdb.per_log_lidar_timestamps_index[log_id]

    for ts in mmcv.track_iter_progress(cam_timestamps):
        cam_ring_fpath = [loader.get_closest_img_fpath(
            log_id, cam_name, ts
        ) for cam_name in CAM_NAMES]
        lidar_fpath = loader.get_closest_lidar_fpath(log_id, ts)

        # If bad sensor synchronization, discard the sample
        if None in cam_ring_fpath or lidar_fpath is None:
            discarded += 1
            continue

        cams = {}
        for i, cam_name in enumerate(CAM_NAMES):
            pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)
            cam_timestamp_ns = int(cam_ring_fpath[i].stem)
            cam_city_SE3_ego = loader.get_city_SE3_ego(
                log_id, cam_timestamp_ns)
            cams[cam_name] = dict(
                img_fpath=str(cam_ring_fpath[i]),
                intrinsics=pinhole_cam.intrinsics.K,
                extrinsics=pinhole_cam.extrinsics,
                e2g_translation=cam_city_SE3_ego.translation,
                e2g_rotation=cam_city_SE3_ego.rotation,
            )

        city_SE3_ego = loader.get_city_SE3_ego(log_id, int(ts))
        e2g_translation = city_SE3_ego.translation
        e2g_rotation = city_SE3_ego.rotation
        info = dict(
            e2g_translation=e2g_translation,
            e2g_rotation=e2g_rotation,
            cams=cams,
            lidar_path=str(lidar_fpath),
            timestamp=str(ts),
            log_id=log_id,
            token=str(log_id+'_'+str(ts)))

        info["scene_token"] = str(ts)
        info['dataset'] = 'av2'
        gma = get_vec_map(info, ns_map, ns_map_exp, dest_path, e2g_translation, e2g_rotation, pc_range, out_type=output_type, avm=avm, vis=vis)
        info = gma.get_map_ann(pertube_vers)

        samples.append(info)

    return samples, discarded

###     Based on data processing code from the official MapTRv2 code available under MIT License
###     Original code can be found at https://github.com/hustvl/MapTR/blob/maptrv2/tools/maptrv2

import multiprocessing
import time
import mmcv
import logging
from pathlib import Path
from os import path as osp
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from shapely import ops
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
from shapely.strtree import STRtree
import numpy as np
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from scipy.spatial import distance
from .map_reading import get_vec_map
import warnings
from nuscenes.nuscenes import NuScenes

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
                    info_prefix='av2',
                    dest_path=None,
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
    root_path = osp.join(root_path, split)
    if dest_path is None:
        dest_path = root_path

    from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
    loader = AV2SensorDataLoader(Path(root_path), Path(root_path))
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

    results = []
    for log_id in mmcv.track_iter_progress(log_ids):
        result = _get_data_from_logid(
            log_id, loader=loader, data_root=root_path, pertube_vers=pertube_vers, pc_range=pc_range, output_type=output_type,vis=vis)
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


def get_divider(avm):
    from av2.map.lane_segment import LaneMarkType
    divider_list = []
    for ls in avm.get_scenario_lane_segments():
        for bound_type, bound_city in zip([ls.left_mark_type, ls.right_mark_type], [ls.left_lane_boundary, ls.right_lane_boundary]):
            if bound_type not in [LaneMarkType.NONE,]:
                divider_list.append(bound_city.xyz)
    return divider_list


def get_boundary(avm):
    boundary_list = []
    for da in avm.get_scenario_vector_drivable_areas():
        boundary_list.append(da.xyz)
    return boundary_list


def get_ped(avm):
    ped_list = []
    for pc in avm.get_scenario_ped_crossings():
        ped_list.append(pc.polygon)
    return ped_list


def _get_data_from_logid(log_id,
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

    ns = NuScenes(version='v1.0-mini',
                  dataroot='/home/li/Documents/map/data/sets/nuscenes', verbose=True)
    ns_map = NuScenesMap("/home/li/Documents/map/data/sets/nuscenes")
    ns_map_exp = NuScenesMapExplorer(ns_map)

    for ts in cam_timestamps:
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
        gma = get_vec_map(info, ns_map, ns_map_exp, e2g_translation,
                          e2g_rotation, pc_range, avm, out_type=output_type, vis_path=vis_path, vis=vis)
        info = gma.get_map_ann(pertube_vers)

        samples.append(info)

    return samples, discarded


def merge_dividers(divider_list):
    # divider_list: List[np.array(N,3)]
    if len(divider_list) < 2:
        return divider_list
    divider_list_shapely = [LineString(divider) for divider in divider_list]
    poly_dividers = [divider.buffer(1,
                                    cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for divider in divider_list_shapely]
    tree = STRtree(poly_dividers)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(poly_dividers))
    final_pgeom = []
    remain_idx = [i for i in range(len(poly_dividers))]
    for i, pline in enumerate(poly_dividers):
        if i not in remain_idx:
            continue
        remain_idx.pop(remain_idx.index(i))
        final_pgeom.append(divider_list[i])
        for o in tree.query(pline):
            o_idx = index_by_id[id(o)]
            if o_idx not in remain_idx:
                continue
            # remove highly overlap divider
            inter = o.intersection(pline).area
            o_iof = inter / o.area
            p_iof = inter / pline.area
            # if query divider is highly overlapped with latter dividers, just remove it
            if p_iof >= 0.95:
                final_pgeom.pop()
                break
            # if queried divider is highly overlapped with query divider,
            # drop it and just turn to next one.
            if o_iof >= 0.95:
                remain_idx.pop(remain_idx.index(o_idx))
                continue

            pline_se_pts = final_pgeom[-1][[0, -1], :2]  # only on xy
            o_se_pts = divider_list[o_idx][[0, -1], :2]  # only on xy
            four_se_pts = np.concatenate([pline_se_pts, o_se_pts], axis=0)
            dist_mat = distance.cdist(four_se_pts, four_se_pts, 'euclidean')
            for j in range(4):
                dist_mat[j, j] = 100
            index = np.where(dist_mat == 0)[0].tolist()
            if index == [0, 2]:
                # e oline s s pline e
                # +-------+ +-------+
                final_pgeom[-1] = np.concatenate(
                    [np.flip(divider_list[o_idx], axis=0)[:-1], final_pgeom[-1]])
                remain_idx.pop(remain_idx.index(o_idx))
            elif index == [1, 2]:
                # s pline e s oline e
                # +-------+ +-------+
                final_pgeom[-1] = np.concatenate(
                    [final_pgeom[-1][:-1], divider_list[o_idx]])
                remain_idx.pop(remain_idx.index(o_idx))
            elif index == [0, 3]:
                # s oline e s pline e
                # +-------+ +-------+
                final_pgeom[-1] = np.concatenate(
                    [divider_list[o_idx][:-1], final_pgeom[-1]])
                remain_idx.pop(remain_idx.index(o_idx))
            elif index == [1, 3]:
                # s pline e e oline s
                # +-------+ +-------+
                final_pgeom[-1] = np.concatenate(
                    [final_pgeom[-1][:-1], np.flip(divider_list[o_idx], axis=0)])
                remain_idx.pop(remain_idx.index(o_idx))
            elif len(index) > 2:
                remain_idx.pop(remain_idx.index(o_idx))

    return final_pgeom


def extract_local_boundary(avm, ego_SE3_city, patch_box, patch_angle, patch_size, pt):
    boundary_list = []
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    for da in avm.get_scenario_vector_drivable_areas():
        boundary_list.append(da.xyz)

    polygon_list = []
    for da in boundary_list:
        exterior_coords = da
        interiors = []
        polygon = Polygon(exterior_coords, interiors)
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                if new_polygon.geom_type == 'Polygon':
                    if not new_polygon.is_valid:
                        continue
                    new_polygon = proc_polygon(new_polygon, ego_SE3_city)
                    if not new_polygon.is_valid:
                        continue
                elif new_polygon.geom_type == 'MultiPolygon':
                    polygons = []
                    for single_polygon in new_polygon.geoms:
                        if not single_polygon.is_valid or single_polygon.is_empty:
                            continue
                        new_single_polygon = proc_polygon(
                            single_polygon, ego_SE3_city)
                        if not new_single_polygon.is_valid:
                            continue
                        polygons.append(new_single_polygon)
                    if len(polygons) == 0:
                        continue
                    new_polygon = MultiPolygon(polygons)
                    if not new_polygon.is_valid:
                        continue
                else:
                    raise ValueError(
                        '{} is not valid'.format(new_polygon.geom_type))

                if new_polygon.geom_type == 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                polygon_list.append(new_polygon)

    union_segments = ops.unary_union(polygon_list)
    max_x = patch_size[1] / 2
    max_y = patch_size[0] / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    exteriors = []
    interiors = []
    if union_segments.geom_type != 'MultiPolygon':
        union_segments = MultiPolygon([union_segments])
    for poly in union_segments.geoms:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)

    results = []
    for ext in exteriors:
        if ext.is_ccw:
            ext.coords = list(ext.coords)[::-1]
        lines = ext.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)

    for inter in interiors:
        if not inter.is_ccw:
            inter.coords = list(inter.coords)[::-1]
        lines = inter.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)

    boundary_lines = []
    for line in results:
        if not line.is_empty:
            if line.geom_type == 'MultiLineString':
                for single_line in line.geoms:
                    boundary_lines.append(np.array(single_line.coords))
            elif line.geom_type == 'LineString':
                boundary_lines.append(np.array(line.coords))
            else:
                raise NotImplementedError
    return boundary_lines

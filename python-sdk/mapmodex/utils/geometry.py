import copy
import math
import random
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import linemerge, unary_union
from shapely import affinity


def check_divider_common(divider, divider_list):
    for ind, d in enumerate(divider_list):
        merged_line = merge_polylines_if_same_track(divider, d)
        if merged_line:
            divider_list.pop(ind)
            return [merged_line, divider_list]
        
    return None

# Function to move polygon along polyline
def move_polygon_along_polyline(polygon, polyline, move_distance):
    # Find the nearest point on the polyline to the polygon centroid
    centroid = polygon.centroid
    nearest_point = polyline.interpolate(polyline.project(centroid))
    
    # Get points at specified distances forward and backward along the polyline
    forward_point = polyline.interpolate(polyline.project(nearest_point) + move_distance)
    backward_point = polyline.interpolate(polyline.project(nearest_point) - move_distance)
    
    # Calculate the translation needed for forward movement
    dx_forward = forward_point.x - nearest_point.x
    dy_forward = forward_point.y - nearest_point.y
    moved_polygon_forward = affinity.translate(polygon, xoff=dx_forward, yoff=dy_forward)

    # Calculate the translation needed for backward movement
    dx_backward = backward_point.x - centroid.x
    dy_backward = backward_point.y - centroid.y
    moved_polygon_backward = affinity.translate(polygon, xoff=dx_backward, yoff=dy_backward)

    return moved_polygon_forward, moved_polygon_backward

def merge_polylines_if_same_track(line1, line2):
    # create LineString objects
    # line1 = LineString(polyline1)
    # line2 = LineString(polyline2)
    
    # Check if they intersect or overlap
    if line1.intersects(line2):
        # The intersection may be a point, a line segment, or multiple geometric objects.
        intersection = line1.intersection(line2)
        
        # Check if the intersection is a line segment (meaning they are on the same path)
        if intersection.geom_type == 'LineString':
            # Merge two polylines
            merged_line = unary_union([line1, line2])#, grid_size=1)
            if merged_line.geom_type == 'MultiLineString':
                merged_line = linemerge([line for line in merged_line.geoms()])#, grid_size=1)
            
            return merged_line
    return None

def one_type_line_geom_to_instances(line_geom):
    line_instances = []

    for line in line_geom:
        if not line.is_empty:
            if line.geom_type == 'MultiLineString':
                for single_line in line.geoms:
                    line_instances.append(single_line)
            elif line.geom_type == 'LineString':
                line_instances.append(line)
            elif line.geom_type == 'LinearRing':
                line_instances.append(line)
            else:
                raise NotImplementedError
    return line_instances

def move_geom(centerline_center, polyline, distance):
    polyline_center = Point(polyline.centroid)
    try:
        direction = np.array([polyline_center.x - centerline_center.x, polyline_center.y - centerline_center.y])
    except:
        print('not able to move')
        return polyline
    
    mv = direction / np.max(abs(direction))* distance
    moved_polyline = affinity.translate(polyline, xoff=mv[0], yoff=mv[1])

    return moved_polyline

def creat_boundray(bd):
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

def zoom_patch_by_layers(layer_name, num_layer_elements, tran_args, vect_dict, _zoom_grid, patch_box, fix_corner, _warping):
    times = math.floor(num_layer_elements[layer_name] * tran_args.wid_lan[1])
    index_list = random.choices([i for i in range(num_layer_elements[layer_name])], k=times)

    for ind, ele in enumerate(vect_dict[layer_name]):
        if ind in index_list:
            widen_area = (ele[:, 0].min(), ele[:, 1].min(),ele[:, 0].max(), ele[:, 1].max())
            g_xv, g_yv = _zoom_grid(patch_box, tran_args.wid_lan[2], widen_area)

            for key in vect_dict.keys():
                if len(vect_dict[key]):
                    for ind, ins in enumerate(vect_dict[key]):
                        ins = fix_corner(ins, [0, 0, patch_box[2], patch_box[3]])
                        vect_dict[key][ind] = _warping(ins, g_xv, g_yv)

def delete_duplicate_centerline(centerline_dict, lane_dict):
    cl_list = [copy.deepcopy(cl_dic) for cl_dic in centerline_dict.values()]
    
    for ind, cl_dic in enumerate(cl_list):
        cl_lane_list = set(cl_dic['lane_token'])
        next_ind = ind + 1
        while next_ind < len(cl_list):
            tem_cl_lane_list = set(cl_list[next_ind]['lane_token'])
            if cl_lane_list == tem_cl_lane_list:
                centerline_dict.pop(cl_dic['token'])
                for lane_token in cl_dic['lane_token']:
                    lane_dict[lane_token]['centerline_token'].remove(cl_dic['token'])
                break
            next_ind += 1
            
    return centerline_dict, lane_dict

def remove_polyline_overlap(line):
    """
    Remove overlapping segments from a polyline, represented by a list of (x, y) coordinates.
    
    Args:
    - coords: list of tuples representing (x, y) coordinates of the polyline.

    Returns:
    - list: New polyline without overlapping segments.
    """
    coords = list(line.coords)
    
    if len(coords) < 2:
        return coords  # A polyline with fewer than 2 points cannot overlap

    line_segments = []

    # Create line segments
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        line_segments.append(segment)
    
    non_overlapping_segments = []
    # seen_segments = set()

    # Iterate through segments and keep only non-overlapping ones
    for i, segment1 in enumerate(line_segments):
        is_overlapping = False
        for j, segment2 in enumerate(line_segments):
            if i != j and segment1.overlaps(segment2):
                is_overlapping = True
                break
        if not is_overlapping:
            non_overlapping_segments.append(segment1)
            # seen_segments.add(segment1)

    # Convert non-overlapping segments back into a polyline
    new_coords = []
    for segment in non_overlapping_segments:
        if not new_coords:
            new_coords.extend(list(segment.coords))
        else:
            new_coords.extend(list(segment.coords)[1:])  # Avoid duplicating points

    new_line = LineString(new_coords)
    
    return new_line
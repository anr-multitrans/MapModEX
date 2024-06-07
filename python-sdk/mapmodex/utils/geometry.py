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

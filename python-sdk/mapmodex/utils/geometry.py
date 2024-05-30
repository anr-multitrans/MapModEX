from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union


def check_divider_common(divider, divider_list):
    for ind, d in enumerate(divider_list):
        merged_line = merge_polylines_if_same_track(divider, d)
        if merged_line:
            divider_list.pop(ind)
            return [merged_line, divider_list]
        
    return None

def merge_polylines_if_same_track(line1, line2):
    # 创建 LineString 对象
    # line1 = LineString(polyline1)
    # line2 = LineString(polyline2)
    
    # 检查它们是否相交或者重叠
    if line1.intersects(line2):
        # 交集可能是一个点、一段线段或者多个几何对象
        intersection = line1.intersection(line2)
        
        # 检查交集是否为一条线段（意味着它们在同一条路径上）
        if intersection.geom_type == 'LineString':
            # 合并两个折线
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
            else:
                raise NotImplementedError
    return line_instances

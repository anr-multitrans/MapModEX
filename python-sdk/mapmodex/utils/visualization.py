import math
import os
import pickle
from typing import List, Optional, Tuple

import descartes
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Arrow, Rectangle
from nuscenes.map_expansion.bitmap import BitMap
from .utilities import *
# Recommended style to use as the plots will show grids.
plt.style.use('seaborn-whitegrid')


colors_plt = {'divider': '#808000', 'ped_crossing': '#000080',
              'boundary': '#008000', 'centerline': 'mistyrose', 'agent': 'r', 'lane': 'b'}


class RenderMap:

    def __init__(self,
                 info,
                 vector_map,
                 vis_path: str,
                 vis_switch=True,
                 vis_show=False):
        """
        :param map_api: NuScenesMap database class.
        :param color_map: Color map.
        """
        # Mutable default argument.
        self.color_map = dict(drivable_area='#a6cee3',
                              road_segment='#1f78b4',
                              road_block='#b2df8a',
                              lane='#33a02c',
                              ped_crossing='#fb9a99',
                              walkway='#e31a1c',
                              stop_line='#fdbf6f',
                              carpark_area='#ff7f00',
                              road_divider='#cab2d6',
                              lane_divider='#6a3d9a',
                              traffic_light='#7e772e')

        self.info = info
        self.map_api = vector_map.nusc_map
        self.map_exploer = vector_map.map_explorer
        self.switch = vis_switch
        self.show = vis_show
        self.save = os.path.join(vis_path, info['scene_token'], info['token'])
        self.canvas_max_x = self.map_api.canvas_edge[0]
        self.canvas_min_x = 0
        self.canvas_max_y = self.map_api.canvas_edge[1]
        self.canvas_min_y = 0

    def vis_patch(self, patch_box):
        if self.switch:
            patch_coords = patch_box_2_coords(patch_box)
            fig, ax = self.map_api.render_map_patch(
                patch_coords, ['road_segment', 'lane'], figsize=(10, 10))

            if self.save is not None:
                check_path(self.save)
                map_path = os.path.join(self.save, 'org.png')
                plt.savefig(map_path, bbox_inches='tight',
                            format='png', dpi=1200)

                plt.close()

                fig, ax = self.map_api.render_map_patch(
                    patch_coords, ['road_segment'], figsize=(10, 10))
                map_path = os.path.join(self.save, 'road_segment.png')
                plt.savefig(map_path, bbox_inches='tight',
                            format='png', dpi=1200)
                plt.close()

                fig, ax = self.map_api.render_map_patch(
                    patch_coords, ['lane'], figsize=(10, 10))
                map_path = os.path.join(self.save, 'lane.png')
                plt.savefig(map_path, bbox_inches='tight',
                            format='png', dpi=1200)

            if self.show:
                plt.show()

            plt.close()

    def _render_layer(self, map_anns, ax: Axes, layer_name: str, alpha: float, tokens: List[str] = None) -> None:
        """
        Wrapper method that renders individual layers on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: Optional list of tokens to render. None means all tokens are rendered.
        """
        if layer_name in self.map_api.non_geometric_polygon_layers:
            self._render_polygon_layer(map_anns, ax, layer_name, alpha, tokens)
        elif layer_name in self.map_api.non_geometric_line_layers:
            self._render_line_layer(map_anns, ax, layer_name, alpha, tokens)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def _render_polygon_layer(self, map_anns, ax: Axes, layer_name: str, alpha: float, tokens: List[str] = None) -> None:
        """
        Renders an individual non-geometric polygon layer on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: Optional list of tokens to render. None means all tokens are rendered.
        """
        if layer_name not in self.map_api.lookup_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        first_time = True
        records = getattr(self.map_api, layer_name)
        if tokens is not None:
            records = [r for r in records if r['token'] in tokens]
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_api.extract_polygon(
                    polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    if first_time:
                        label = layer_name
                        first_time = False
                    else:
                        label = None
                    ax.add_patch(descartes.PolygonPatch(polygon, fc=self.color_map[layer_name], alpha=alpha,
                                                        label=label))
        else:
            for polygon in map_anns:

                if first_time:
                    label = layer_name
                    first_time = False
                else:
                    label = None

                ax.add_patch(descartes.PolygonPatch(polygon, fc=self.color_map[layer_name], alpha=alpha,
                                                    label=label))

    def _render_line_layer(self, map_anns, ax: Axes, layer_name: str, alpha: float, tokens: List[str] = None) -> None:
        """
        Renders an individual non-geometric line layer on an axis.
        :param ax: The matplotlib axes where the layer will get rendered.
        :param layer_name: Name of the layer that we are interested in.
        :param alpha: The opacity of the layer to be rendered.
        :param tokens: Optional list of tokens to render. None means all tokens are rendered.
        """
        if layer_name not in self.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        first_time = True
        if tokens is not None:
            records = [r for r in records if r['token'] in tokens]
        for line in map_anns:
            if first_time:
                label = layer_name
                first_time = False
            else:
                label = None
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            if layer_name == 'traffic_light':
                # Draws an arrow with the physical traffic light as the starting point, pointing to the direction on
                # where the traffic light points.
                ax.add_patch(Arrow(xs[0], ys[0], xs[1]-xs[0], ys[1]-ys[0], color=self.color_map[layer_name],
                                   label=label))
            else:
                ax.plot(
                    xs, ys, color=self.color_map[layer_name], alpha=alpha, label=label)

    def get_map_mask(self,
                     geom,
                     patch_box: Optional[Tuple[float, float, float, float]],
                     patch_angle: float,
                     layer_names: List[str] = None,
                     canvas_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Return list of map mask layers of the specified patch.
        :param patch_box: Patch box defined as [x_center, y_center, height, width]. If None, this plots the entire map.
        :param patch_angle: Patch orientation in degrees. North-facing corresponds to 0.
        :param layer_names: A list of layer names to be extracted, or None for all non-geometric layers.
        :param canvas_size: Size of the output mask (h, w). If None, we use the default resolution of 10px/m.
        :return: Stacked numpy array of size [c x h x w] with c channels and the same width/height as the canvas.
        """
        # For some combination of parameters, we need to know the size of the current map.
        if self.map_api.map_name == 'singapore-onenorth':
            map_dims = [1585.6, 2025.0]
        elif self.map_api.map_name == 'singapore-hollandvillage':
            map_dims = [2808.3, 2922.9]
        elif self.map_api.map_name == 'singapore-queenstown':
            map_dims = [3228.6, 3687.1]
        elif self.map_api.map_name == 'boston-seaport':
            map_dims = [2979.5, 2118.1]
        else:
            raise Exception('Error: Invalid map!')

        # If None, return the entire map.
        if patch_box is None:
            patch_box = [map_dims[0] / 2, map_dims[1] /
                         2, map_dims[1], map_dims[0]]

        # If None, return all geometric layers.
        if layer_names is None:
            layer_names = self.map_api.non_geometric_layers

        # If None, return the specified patch in the original scale of 10px/m.
        if canvas_size is None:
            map_scale = 10
            canvas_size = np.array((patch_box[2], patch_box[3])) * map_scale
            canvas_size = tuple(np.round(canvas_size).astype(np.int32))

        # Get geometry of each layer.
        map_geom = [kv for kv in geom.items()]

        # Convert geometry of each layer into mask and stack them into a numpy tensor.
        # Convert the patch box from global coordinates to local coordinates by setting the center to (0, 0).
        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        map_mask = self.map_exploer.map_geom_to_mask(
            map_geom, local_box, canvas_size)
        assert np.all(map_mask.shape[1:] == canvas_size)

        return map_mask

    def render_map_mask(self,
                        geom,
                        patch_box: Tuple[float, float, float, float],
                        patch_angle=0,  # float,
                        layer_names=None,  # List[str],
                        canvas_size=(1000, 1000),  # Tuple[int, int],
                        figsize=(12, 12),  # Tuple[int, int],
                        n_row: int = 3,
                        version=None) -> Tuple[Figure, List[Axes]]:
        """
        Render map mask of the patch specified by patch_box and patch_angle.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :param layer_names: A list of layer names to be extracted.
        :param canvas_size: Size of the output mask (h, w).
        :param figsize: Size of the figure.
        :param n_row: Number of rows with plots.
        :return: The matplotlib figure and a list of axes of the rendered layers.
        """
        if self.switch:
            if layer_names is None:
                layer_names = self.map_api.non_geometric_layers

            map_mask = self.get_map_mask(
                geom, patch_box, patch_angle, layer_names, canvas_size)

            # If no canvas_size is specified, retrieve the default from the output of get_map_mask.
            if canvas_size is None:
                canvas_size = map_mask.shape[1:]

            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, canvas_size[1])
            ax.set_ylim(0, canvas_size[0])

            n_col = math.ceil(len(map_mask) / n_row)
            gs = gridspec.GridSpec(n_row, n_col)
            gs.update(wspace=0.025, hspace=0.05)
            for i in range(len(map_mask)):
                r = i // n_col
                c = i - r * n_col
                subax = plt.subplot(gs[r, c])
                subax.imshow(map_mask[i], origin='lower')
                subax.text(canvas_size[0] * 0.5,
                           canvas_size[1] * 1.1, layer_names[i])
                subax.grid(False)

            if self.save is not None:
                check_path(self.save)
                plt.savefig(os.path.join(self.save, version))

            if self.show:
                plt.show()

            plt.close()

    def vis_contours(self, contours, patch_box, map_version):
        if self.switch:

            plt.figure(figsize=(2, 4))
            # plt.figure
            plt.xlim(-patch_box[3]/2, patch_box[3]/2)
            plt.ylim(-patch_box[2]/2, patch_box[2]/2)
            plt.axis('off')
            
            for pred_label_3d in contours.keys():
                if pred_label_3d in self.info['order'] and len(contours[pred_label_3d]):
                    for pred_pts_3d in contours[pred_label_3d]:
                        pts_x = pred_pts_3d[:, 0]
                        pts_y = pred_pts_3d[:, 1]
                        plt.plot(
                            pts_x, pts_y, color=colors_plt[pred_label_3d], linewidth=1, alpha=0.8, zorder=-1)
                        plt.scatter(
                            pts_x, pts_y, color=colors_plt[pred_label_3d], s=1, alpha=0.8, zorder=-1)

            if self.save is not None:
                check_path(self.save)
                map_path = os.path.join(self.save, map_version+'.png')
                plt.savefig(map_path, bbox_inches='tight',
                            format='png', dpi=1200)

            if self.show:
                plt.show()

            plt.close()

    def vis_polygones(self):
        pass


def vis_contours_local(contours, show_layers = None, patch_box=[0, 0, 62, 32], save_path=None, show=False):
    plt.figure(figsize=(2, 4))
    plt.xlim(-patch_box[3]/2, patch_box[3]/2)
    plt.ylim(-patch_box[2]/2, patch_box[2]/2)
    plt.axis('off')
    
    if show_layers is None:
        show_layers = [key for key in contours.keys()]
        
    for pred_label_3d in show_layers:
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
        map_path = os.path.join(save_path, 'map_ex.png')
        plt.savefig(map_path, bbox_inches='tight',
                    format='png', dpi=1200)

    if show:
        plt.show()

    plt.close()


def show_img(X):
    plt.imshow(X, interpolation='nearest')
    plt.show()
    plt.close()

def show_geom(new_shape):
    fig, axs = plt.subplots()
    axs.set_aspect('equal', 'datalim')  
    
    if new_shape.geom_type in ["MultiPolygon", "MultiLineString"]:
        for geom in new_shape.geoms:    
            xs, ys = geom.exterior.xy    
            axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
    if new_shape.geom_type in ["Polygon", "LineString", "Point"]:
        xs, ys = geom.exterior.xy    
        axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
    else:
        os.exit("wrong geom type: ", new_shape.geom_type)
        
    plt.show()

if __name__ == '__main__':
    map_path = 'MapTRV2Local/tools/maptrv2/map_perturbation/pt_map/cc8c0bf57f984915a77078b10eb33198/4f545737bf3347fbbc9af60b0be9a963/perturbated_map_json_0/maps/expansion/pt_patch.pkl'
    save_map_path = 'MapTRV2Local/tools/maptrv2/map_perturbation/pt_map/cc8c0bf57f984915a77078b10eb33198/4f545737bf3347fbbc9af60b0be9a963/perturbated_map_json_0/maps/expansion'

    with open(map_path, 'rb') as f:
        ret_di = pickle.load(f)

    vis_contours_local(ret_di, save_path=save_map_path)
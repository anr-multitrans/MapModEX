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
from matplotlib.widgets import Button, TextBox
import tkinter as tk
from tkinter import messagebox
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon, box
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shapely.errors import TopologicalError
from shapely.validation import make_valid


from .utilities import *
# Recommended style to use as the plots will show grids.
plt.style.use('seaborn-whitegrid')


colors_plt = {'divider': '#808000', 'ped_crossing': '#000080',
              'boundary': '#008000', 'centerline': 'mistyrose', 'agent': 'red', 'lane': 'blue'}
linewidth_plt = {'divider': 1, 'ped_crossing': 1,
              'boundary': 1, 'centerline': 2, 'agent': 1, 'lane': 1}
clipping_box = box(-15, -30, 15, 30)

def plot_geometry(ax, geometry, index=None, color='blue', linewidth=1, fontsize=12):
    try:
        clipped_geometry = geometry.intersection(clipping_box)
    except TopologicalError:
        # Fix the invalid geometry
        valid_geometry = make_valid(geometry)
        clipped_geometry = valid_geometry.intersection(clipping_box)
    
    if clipped_geometry.is_empty:
        return
    
    if isinstance(clipped_geometry, Point):
        ax.plot(clipped_geometry.x, clipped_geometry.y, 'o', color=color, label=f'{index}' if index is not None else '', linewidth=linewidth)
        if index is not None:
            ax.text(clipped_geometry.x, clipped_geometry.y, f'{index}', fontsize=fontsize, fontweight='bold', ha='right')
    elif isinstance(clipped_geometry, LineString):
        x, y = clipped_geometry.xy
        ax.plot(x, y, color=color, label=f'{index}' if index is not None else '', linewidth=linewidth)
        if index is not None:
            ax.text(min(max(x[0], clipping_box.bounds[0]), clipping_box.bounds[2]),
                    min(max(y[0], clipping_box.bounds[1]), clipping_box.bounds[3]),
                    f'{index}', fontsize=fontsize, fontweight='bold', ha='right')
    elif isinstance(clipped_geometry, Polygon):
        x, y = clipped_geometry.exterior.xy
        ax.plot(x, y, color=color, label=f'{index}' if index is not None else '', linewidth=linewidth)
        if index is not None:
            ax.text(min(max(x[0], clipping_box.bounds[0]), clipping_box.bounds[2]),
                    min(max(y[0], clipping_box.bounds[1]), clipping_box.bounds[3]),
                    f'{index}', fontsize=fontsize, fontweight='bold', ha='right')
    elif isinstance(clipped_geometry, MultiLineString):
        for linestring in clipped_geometry:
            x, y = linestring.xy
            ax.plot(x, y, color=color, label=f'{index}' if index is not None else '', linewidth=linewidth)
            if index is not None:
                ax.text(min(max(x[0], clipping_box.bounds[0]), clipping_box.bounds[2]),
                        min(max(y[0], clipping_box.bounds[1]), clipping_box.bounds[3]),
                        f'{index}', fontsize=fontsize, fontweight='bold', ha='right')
    elif isinstance(clipped_geometry, MultiPolygon):
        for polygon in clipped_geometry:
            x, y = polygon.exterior.xy
            ax.plot(x, y, color=color, label=f'{index}' if index is not None else '', linewidth=linewidth)
            if index is not None:
                ax.text(min(max(x[0], clipping_box.bounds[0]), clipping_box.bounds[2]),
                        min(max(y[0], clipping_box.bounds[1]), clipping_box.bounds[3]),
                        f'{index}', fontsize=fontsize, fontweight='bold', ha='right')

def update_canvas(figure, ax, interactive_geometries, static_geometries):
    ax.clear()
    ax.set_xlim(-15, 15)
    ax.set_ylim(-30, 30)
    for st_layer, geometrys in static_geometries.items():
        for geometry in geometrys:
            if isinstance(geometry, dict):
                plot_geometry(ax, geometry['geom'], color=colors_plt[st_layer])
            else:
                plot_geometry(ax, geometry, color=colors_plt[st_layer])
    
    for in_layer, geometrys in interactive_geometries.items():
        for i, geometry in enumerate(geometrys):
            if isinstance(geometry, dict):
                plot_geometry(ax, geometry['geom'], i, color=colors_plt[in_layer], linewidth=3, fontsize=32)
            else:
                plot_geometry(ax, geometry, i, color=colors_plt[in_layer], linewidth=3, fontsize=32)
    
    ax.relim()
    ax.autoscale_view()
    figure.canvas.draw()

def on_choose(figure, ax, interactive_geometries, static_geometries, input_box, deleted_geometries):
    input_value = input_box.get()
    try:
        indices = list(map(int, input_value.split()))
        indices = sorted(set(indices), reverse=True)
        
        for v in interactive_geometries.values():
            for index in indices:
                if 0 <= index < len(v):
                    deleted_geometries.append(v[index])
                    del v[index]
                else:
                    raise ValueError
        update_canvas(figure, ax, interactive_geometries, static_geometries)
    except ValueError:
        messagebox.showwarning("Invalid input", "Please enter valid integer indices.")

def on_done(root):
    root.quit()
    root.destroy()

def get_bounding_box(geometries):
    minx, miny, maxx, maxy = None, None, None, None
    for geometry in geometries:
        if isinstance(geometry, dict):
            bounds = geometry['geom'].bounds
        else:
            bounds = geometry.bounds
            
        if minx is None or bounds[0] < minx:
            minx = bounds[0]
        if miny is None or bounds[1] < miny:
            miny = bounds[1]
        if maxx is None or bounds[2] > maxx:
            maxx = bounds[2]
        if maxy is None or bounds[3] > maxy:
            maxy = bounds[3]
    return minx, miny, maxx, maxy

def geometry_manager(geometries_dict, interactive_layer, static_layers): #TODO add a vect_dict option
    all_geometries = []
    
    interactive_geometries = {}
    if isinstance(geometries_dict, dict):
        interactive_geometries[interactive_layer] = [geom for geom in geometries_dict[interactive_layer].values()]
    else:
        interactive_geometries[interactive_layer] = geometries_dict[interactive_layer]
        
    all_geometries += interactive_geometries[interactive_layer]
    
    static_geometries = {}
    for layer in static_layers:
        if isinstance(geometries_dict, dict):
            if layer == 'lane':
                static_geometries[layer] = [geom for geom in geometries_dict[layer].values() if geom['from'] != 'lane_connector']
            else:
                static_geometries[layer] = [geom for geom in geometries_dict[layer].values()]
        else:
            static_geometries[layer] = geometries_dict[layer]
    
        all_geometries += static_geometries[layer]
    
    deleted_geometries = []
    
    minx, miny, maxx, maxy = get_bounding_box(all_geometries)
    
    root = tk.Tk()
    root.title("Geometry Manager")
    
    figure, ax = plt.subplots(figsize=(8, 6))
    canvas = FigureCanvasTkAgg(figure, root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    input_frame = tk.Frame(root)
    input_frame.pack(pady=10)
    
    input_label = tk.Label(input_frame, text="Enter indices to remove (space-separated):", font=("Arial", 14, "bold"))
    input_label.pack(side=tk.LEFT)
    
    input_box = tk.Entry(input_frame, font=("Arial", 14, "bold"))
    input_box.pack(side=tk.LEFT)
    
    choose_button = tk.Button(input_frame, text="Choose", font=("Arial", 14, "bold"),
                              command=lambda: on_choose(figure, ax, interactive_geometries, static_geometries, input_box, deleted_geometries))
    choose_button.pack(side=tk.LEFT, padx=5)
    
    done_button = tk.Button(root, text="Done", font=("Arial", 14, "bold"), command=lambda: on_done(root))
    done_button.pack(pady=10)
    
    update_canvas(figure, ax, interactive_geometries, static_geometries)
    
    root.geometry(f"{int((maxx - minx) * 60 + 200)}x{int((maxy - miny) * 60 + 200)}")
    
    root.mainloop()
    
    return deleted_geometries

class RenderMap:

    def __init__(self,
                 info,
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
        self.switch = vis_switch
        self.show = vis_show
        self.save = os.path.join(vis_path, info['scene_token'], info['token'])
        self.canvas_min_x = 0
        self.canvas_min_y = 0
        self.patch_box = [0,0,60,30]

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

    def vis_contours(self, contours, map_version):
        if self.switch:

            plt.figure(figsize=(2, 4))
            plt.xlim(-self.patch_box[3]/2, self.patch_box[3]/2)
            plt.ylim(-self.patch_box[2]/2, self.patch_box[2]/2)
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

import os
import matplotlib.pyplot as plt
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

# Function to add polygons to the plot
def add_geometry(ax, geom, **kwargs):
    if geom.is_empty:
        return
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        ax.fill(x, y, **kwargs)
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom:
            x, y = poly.exterior.xy
            ax.fill(x, y, **kwargs)
    elif geom.geom_type == 'LineString':
        x, y = geom.xy
        ax.plot(x, y, **kwargs)
    elif geom.geom_type == 'MultiLineString':
        for line in geom:
            x, y = line.xy
            ax.plot(x, y, **kwargs)
    else:
        print(f"Warning: Unsupported geometry type {geom.geom_type}")

def show_geoms(geometries):

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each geometry in the list
    for geom in geometries:
        if isinstance(geom, (Polygon, MultiPolygon, LineString, MultiLineString)):
            add_geometry(ax, geom, alpha=0.5)
        else:
            print(f"Warning: Unsupported geometry {type(geom)}")
    
    # Adding titles and labels
    ax.set_title('Polygons and MultiPolygons on One Canvas')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

    # Show plot
    plt.show()

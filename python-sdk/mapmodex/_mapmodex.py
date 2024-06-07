import os
from .scripts import create_nuscenes_infos, create_av2_infos, creat_mme_infos, PerturbParameters


class MapModEX():
    def __init__(self, data_root:str, output_path = None, pc_range=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0]):
        """
        Args:
            data_root (str): path of the map database
            output_path (str, optional): output path. Defaults to None: current address.
            pc_range (list, optional): Patch size (3D):[ymin, xmin, zmin, ymax, xmax, zmax]. Defaults to [-30.0, -15.0, -5.0, 30.0, 15.0, 3.0].
        """
        self.data_root = data_root
        self.pc_range = pc_range
        
        if output_path is None:
            self.output = os.path.join(os.getcwd, 'mapmodex_output')
            if not self.output:
                os.mkdir(self.output)
        else:
            self.output = output_path
            
        self.pt_version_name = 0
        self.pt_version = []
        pt_v = {'pt_name':'org'}
        self._add_pt_version(pt_v)
    
    def _update_attribute(self, attr_name, new_value):
        if hasattr(self, attr_name):
            setattr(self, attr_name, new_value)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr_name}'")     
    
    def _add_pt_version(self, pt_v):
        pt_para = PerturbParameters()
        
        if pt_para.pt_name is None:
            self.pt_version_name += 1
            pt_para.pt_name = 'pt_v_' +str(self.pt_version_name)
        
        for key, val in pt_v.items():
            pt_para.update_attribute(key, val)
    
        self.pt_version.append(pt_para)
           
    def update_pt_version(self, pt_type: list, org = True):
        """Pass the perturbed version with parameters to self

        Args:
            pt_type (list): perturbed versions, each version should be a dict with parameter_names and parameter_values.
            ex. [{'pt_name': 'pt_1', 'add_lan':[1,0.5,None]}]
            org (bool, optional): Whether to output the original image. Defaults to True.
        """
        if not org:
            self.pt_version = []
        
        for pt_v in pt_type:
            self._add_pt_version(pt_v)
            
        
    def mod_nuscenes(self, map_version:list, root_name='nuscenes', output_type=None, vis=False):
        """Perturb the nuScenes map

        Args:
            map_version (list): map versions, ex. ['v1.0-mini']
            root_name (str, optional): Map database folder name. Defaults to 'nuscenes'.
            output_type (str, optional): output tyoe. 'pkl' is the data used for model training, and 'json' is the map data. Defaults to 'json'.
            vis (bool, optional): Whether to visualize. Defaults to False.
        """
        for v_name in map_version:
            print('generating %s - %s' % ('NuScenes', v_name))
            out_put_path = os.path.join(self.output, 'nuscenes_output', v_name)
            create_nuscenes_infos(
                root_path=os.path.join(self.data_root, root_name),
                out_path=out_put_path,
                pertube_vers=self.pt_version,
                version=v_name,
                vis = vis,
                out_type = output_type)
            print('results are saved at: ', out_put_path)
            
    def mod_av2(self, map_version:list, root_name='av2', output_type='json', vis=False):
        """Perturb the argoverse 2 map

        Args:
            map_version (list): map versions, ex. ['test']
            root_name (str, optional): Map database folder name. Defaults to 'av2'.
            output_type (str, optional): output tyoe. 'pkl' is the data used for model training, and 'json' is the map data. Defaults to 'json'.
            vis (bool, optional): Whether to visualize. Defaults to False.
        """
        for v_name in map_version:
            print('generating %s - %s' % ('argoverse 2', v_name))
            out_put_path = os.path.join(self.output, 'av2_output', v_name)
            create_av2_infos(
                root_path=os.path.join(self.data_root, root_name),
                pertube_vers=self.pt_version,
                dest_path=out_put_path,
                split=v_name,
                pc_range=self.pc_range,
                vis=vis,
                output_type=output_type)
            print('results are saved at: ', out_put_path)

    def mod_mme(self, map_version:list, root_name='mme', output_type='json', vis=False): #TODO
        """Perturb the MapModEX map: It inherits the map API of nuScenes. The map name is unified as singapore-onenorth.json.

        Args:
            map_version (list): the perturbation version. ex. ['mme_org', 'mme_add_lane']
            root_name (str, optional): Map database folder name. Defaults to 'MME_map'.
            output_type (str, optional): 'pkl' is the data used for model training, and 'json' is the map data. Defaults to 'json'.
            vis (bool, optional): _description_. Defaults to False.
        """
        print('generating %s' % ('MapModEX-map'))
        out_put_path = os.path.join(self.output, 'mme_output')
        creat_mme_infos(
            root_path=os.path.join(self.data_root, root_name),
            map_version=map_version,
            pertube_vers=self.pt_version,
            out_path=os.path.join(self.output, 'mme_output'),
            pc_range=self.pc_range,
            vis=vis,
            output_type=output_type)
        print('results are saved at: ', out_put_path)

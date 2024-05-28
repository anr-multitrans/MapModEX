import os
from .scripts import create_nuscenes_infos, create_av2_infos_mp, PerturbParameters


class MapModEX():
    def __init__(self, data_root:str, output_path = None):
        self.data_root = data_root
        
        if output_path is None:
            self.output = os.path.join(os.getcwd, 'mapmodex_output')
            if not self.output:
                os.mkdir(self.output)
        else:
            self.output = output_path
            
        self.pt_version_name = 0
        self.pt_version = []
        pt_v = {'pt_name':'org'}
        self.add_pt_version(pt_v)
    
    def _update_attribute(self, attr_name, new_value):
        if hasattr(self, attr_name):
            setattr(self, attr_name, new_value)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr_name}'")     
           
    def update_pt_version(self, pt_type: list, org = True):
        if not org:
            self.pt_version = []
        
        for pt_v in pt_type:
            self.add_pt_version(pt_v)
            
    def add_pt_version(self, pt_v):
        pt_para = PerturbParameters()
        
        if 'pt_name' not in pt_v:
            self.pt_version_name += 1
            pt_v['pt_name'] = 'pt_v_' +str(self.pt_version_name)
        
        for key, val in pt_v.items():
            pt_para.update_attribute(key, val)
    
        self.pt_version.append(pt_para)
        
    def mod_nuscenes(self, map_version:list, root_name='nuscenes', max_sweeps=10, output_type='json', info_prefix='nuscenes', vis=False):
        for v_name in map_version:
            print('generating %s - %s' % ('NuScenes', v_name))
            create_nuscenes_infos(
                root_path=os.path.join(self.data_root, 'nuscenes'),
                out_path=os.path.join(self.output, 'nuscenes'),
                info_prefix=info_prefix,
                pertube_vers=self.pt_version,
                version=v_name,
                max_sweeps=max_sweeps,
                vis = vis,
                out_type = output_type)
            
    def mod_av2(self, map_version:list, root_name='av2', pc_range=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0], output_type='json', info_prefix='av2', vis=False):
            for v_name in map_version:
                print('generating %s - %s' % ('argoverse 2', v_name))
                create_av2_infos_mp(
                    root_path=os.path.join(self.data_root, 'av2'),
                    info_prefix=info_prefix,
                    pertube_vers=self.pt_version,
                    dest_path=os.path.join(self.output, 'av2'),
                    split=v_name,
                    pc_range=pc_range,
                    vis=vis,
                    output_type=output_type)

    def mod_mme(self, mme_path = None):
        if mme_path is None:
            mme_path = self.data_root
            
        pass #TODO


        ann_name = 'annotation'
        map_json_name = 'mme'
        for ind, map_v in enumerate(pertube_vers):
            if map_v is not None:
                ann_name = ann_name + '_' + str(ind)
                map_json_name = map_json_name + '_' + str(ind)

            trans_dic = self.vector_map.gen_vectorized_samples(
                map_geom_org_dic, map_v)

            if self.vis_switch:
                trans_np_dict_4_vis = geom_to_np(trans_dic['map_ins_dict'], 20)
                self.visual.vis_contours(
                    trans_np_dict_4_vis, self.patch_box, ann_name)

            self.info[ann_name] = geom_to_np(trans_dic['map_ins_dict'])
            
            if self.map_path is not None:
                self.info[map_json_name] = vector_to_map_json(
                    trans_dic, self.info, map_json_name, self.map_path)


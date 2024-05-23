import numpy as np
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.agents import reverse_history, add_present_time_to_history
from shapely.geometry import LineString, Point

from ..utils import *


def add_future_to_history(future, history):
    for token, annotation in future.items():
        if token in history:
            history[token] += annotation
        else:
            history[token] = annotation

    return history


def get_nuscenes_trajectory(nuscenes: NuScenes, sample_token, agents, seconds_of_expand=3, type='array'):
    helper = PredictHelper(nuscenes)

    history = helper.get_past_for_sample(
        sample_token, seconds_of_expand, in_agent_frame=False, just_xy=False)
    history = reverse_history(history)

    present_time = helper.get_annotations_for_sample(sample_token)
    history = add_present_time_to_history(present_time, history)

    future = helper.get_future_for_sample(
        sample_token, seconds_of_expand, in_agent_frame=False, just_xy=False)
    history = add_future_to_history(future, history)

    agents_tra = {}
    center = None
    for instance_token, annotations in history.items():
        for agent in agents:
            tra = []
            for i, annotation in enumerate(annotations):
                if agent in annotation['category_name']:
                    location = annotation['translation'][:2]
                    tra.append(location)
                    if i == seconds_of_expand:
                        center = location

            if len(tra):
                agents_tra[instance_token] = {}
                agents_tra[instance_token]['token'] = instance_token
                agents_tra[instance_token]['from'] = agent
                if type == 'array':
                    agents_tra[instance_token]['geom'] = np.array(tra)
                    if center is not None:
                        agents_tra[instance_token]['eco'] = np.array(center)
                else:
                    if len(tra) > 1:
                        agents_tra[instance_token]['geom'] = LineString(
                            x for x in tra)
                    else:
                        agents_tra[instance_token]['geom'] = Point(
                            tra[0][0], tra[0][1])
                    agents_tra[instance_token]['eco'] = Point(center)

    return agents_tra


def add_tra_to_vecmap(agent_trajectory, map_exp, patch_box, patch_angle):
    agent_tra = {}

    for tra in agent_trajectory.values():
        new_geom = valid_geom(tra['eco'], map_exp, patch_box, patch_angle)
        if new_geom is not None:
            tra['geom'] = to_patch_coord(tra['geom'], patch_angle, patch_box[0], patch_box[1])
            agent_tra[tra['token']] = tra

    return agent_tra

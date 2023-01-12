#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle

import numpy as np
import open3d as o3d
from scan2cad_dataset_manage.Module.object_model_map_manager import \
    ObjectModelMapManager

from retrieval_metric.Method.retrieval import getOursRetrievalResult


class MetricManager(object):

    def __init__(self):
        self.retrieval_class_accuracy_list = []
        self.scan2cad_chamfer_dist_list = []
        self.retrieval_chamfer_dist_list = []
        self.trans_error_list = []
        self.rotate_error_list = []
        self.scale_error_list = []

        scannet_object_dataset_folder_path = "/home/chli/chLi/ScanNet/objects/"
        shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
        object_model_map_dataset_folder_path = "/home/chli/chLi/Scan2CAD/object_model_maps/"

        self.object_model_map_manager = ObjectModelMapManager(
            scannet_object_dataset_folder_path, shapenet_dataset_folder_path,
            object_model_map_dataset_folder_path)

        print("[INFO][MetricManager::__init__]")
        print("\t start loading uniform_feature_dict...")
        uniform_feature_file_path = "/home/chli/chLi/ShapeNet/uniform_feature/uniform_feature.pkl"
        with open(uniform_feature_file_path, 'rb') as f:
            self.uniform_feature_dict = pickle.load(f)
        return

    def addObjectRetrievalResult(self,
                                 scannet_scene_name,
                                 object_file_name,
                                 print_progress=False):
        shapenet_model_dict = self.object_model_map_manager.getShapeNetModelDict(
            scannet_scene_name, object_file_name)

        object_points, retrieval_cad_mesh = getOursRetrievalResult(
            shapenet_model_dict, self.uniform_feature_dict, print_progress)
        return True

    def addSceneRetrievalResult(self,
                                scannet_scene_name,
                                scene_idx_str,
                                scene_num_str,
                                print_progress=False):
        object_file_name_list = self.object_model_map_manager.getObjectFileNameList(
            scannet_scene_name)

        object_num_str = str(len(object_file_name_list))

        for i, object_file_name in enumerate(object_file_name_list):
            if print_progress:
                print("[INFO][MetricManager::addSceneRetrievalResult]")
                print(
                    "\t start add retrieval results for all objects in scene ["
                    + scannet_scene_name + "], scene " + scene_idx_str + "/" +
                    scene_num_str + "\tobject " + str(i + 1) + "/" +
                    object_num_str + "...")

            self.addObjectRetrievalResult(scannet_scene_name, object_file_name,
                                          print_progress)
        return True

    def getAllMetric(self, print_progress=False):
        scene_num_str = str(len(self.object_model_map_manager.scene_name_list))
        for i, scannet_scene_name in enumerate(
                self.object_model_map_manager.scene_name_list):
            self.addSceneRetrievalResult(scannet_scene_name, str(i + 1),
                                         scene_num_str, print_progress)
        return True

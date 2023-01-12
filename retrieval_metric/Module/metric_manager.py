#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from tqdm import tqdm

from scan2cad_dataset_manage.Module.object_model_map_manager import ObjectModelMapManager


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
        return

    def addObjectRetrievalResult(self, object_file_name):
        return True

    def addSceneRetrievalResult(self,
                                scannet_scene_name,
                                print_progress=False):
        object_file_name_list = self.object_model_map_manager.getObjectFileNameList(
            scannet_scene_name)

        for_data = object_file_name_list
        if print_progress:
            print("[INFO][MetricManager::addSceneRetrievalResult]")
            print("\t start add retrieval results for all objects in scene [" +
                  scannet_scene_name + "]...")
            for_data = tqdm(for_data)
        for object_file_name in for_data:
            self.addObjectRetrievalResult(object_file_name)
        return True

    def getAllMetric(self, print_progress=False):
        for scannet_scene_name in self.object_model_map_manager.scene_name_list:
            self.addSceneRetrievalResult(scannet_scene_name, print_progress)
        return True

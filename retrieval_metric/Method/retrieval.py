#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from global_to_patch_retrieval.Method.feature import getPointsFeature
from global_to_patch_retrieval.Method.retrieval import getObjectRetrievalResult
from points_shape_detect.Method.trans import normalizePointArray


def getOursRetrievalResult(object_pcd,
                           uniform_feature_dict,
                           print_progress=False):
    cad_model_file_path_list = uniform_feature_dict[
        'shapenet_model_file_path_list']
    cad_feature_array = uniform_feature_dict['feature_array']
    cad_mask_array = uniform_feature_dict['mask_array']

    points = np.array(object_pcd.points)
    object_feature, object_mask = getPointsFeature(points, False)

    retrieval_cad_model_file_path = getObjectRetrievalResult(
        object_feature, object_mask, cad_feature_array, cad_mask_array,
        cad_model_file_path_list, print_progress)

    retrieval_cad_mesh = o3d.io.read_triangle_mesh(
        retrieval_cad_model_file_path)
    points = np.array(retrieval_cad_mesh.vertices)
    points = normalizePointArray(points)
    retrieval_cad_mesh.vertices = o3d.utility.Vector3dVector(points)

    retrieval_cad_mesh.compute_triangle_normals()

    #  o3d.visualization.draw_geometries(
    #  [object_pcd, retrieval_cad_mesh])

    return retrieval_cad_mesh, retrieval_cad_model_file_path

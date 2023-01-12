#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d


def getClassAccuracy(gt_mesh_file_path, retrieval_mesh_file_path):
    gt_class = gt_mesh_file_path.split("ShapeNetCore.v2/")[1].split("/")[0]
    retrieval_class = retrieval_mesh_file_path.split(
        "ShapeNetCore.v2/")[1].split("/")[0]

    if gt_class == retrieval_class:
        return 1
    return 0


def getChamferDistance(pcd_1, pcd_2):
    dist_1_array = np.array(pcd_1.compute_point_cloud_distance(pcd_2))
    dist_2_array = np.array(pcd_2.compute_point_cloud_distance(pcd_1))

    dist_1 = np.linalg.norm(dist_1_array)
    dist_2 = np.linalg.norm(dist_2_array)

    chamfer_distance = dist_1 * dist_1 / dist_1_array.shape[
        0] + dist_2 * dist_2 / dist_2_array.shape[0]
    return chamfer_distance


def getTransError(pcd_1, pcd_2):
    center_1 = pcd_1.get_center()
    center_2 = pcd_2.get_center()
    center_dist = np.linalg.norm(center_1 - center_2)
    return center_dist


def getRotateError(pcd_1, pcd_2):
    obb_1 = pcd_1.get_oriented_bounding_box()
    obb_2 = pcd_2.get_oriented_bounding_box()

    R_1 = obb_1.R
    R_2 = obb_2.R

    direction_11 = np.array([1, 0, 0]) @ R_1
    direction_12 = np.array([0, 1, 0]) @ R_1
    direction_21 = np.array([1, 0, 0]) @ R_2
    direction_22 = np.array([0, 1, 0]) @ R_2

    direction_dist_11 = np.min([
        np.linalg.norm(direction_11 - direction_21),
        np.linalg.norm(direction_11 + direction_21)
    ])
    direction_dist_12 = np.min([
        np.linalg.norm(direction_11 - direction_22),
        np.linalg.norm(direction_11 + direction_22)
    ])
    direction_dist_21 = np.min([
        np.linalg.norm(direction_12 - direction_21),
        np.linalg.norm(direction_12 + direction_21)
    ])
    direction_dist_22 = np.min([
        np.linalg.norm(direction_12 - direction_22),
        np.linalg.norm(direction_12 + direction_22)
    ])

    direction_dist = np.min([
        direction_dist_11 + direction_dist_22,
        direction_dist_12 + direction_dist_21
    ]) / 2.0
    cos_error = np.abs(1 - direction_dist * direction_dist / 2.0)
    angle_error = np.arccos(cos_error) * 180.0 / np.pi
    return angle_error


def getScaleError(pcd_1, pcd_2):
    max_point_1 = pcd_1.get_max_bound()
    min_point_1 = pcd_1.get_min_bound()
    max_point_2 = pcd_2.get_max_bound()
    min_point_2 = pcd_2.get_min_bound()
    scale_1 = max_point_1 - min_point_1
    scale_2 = max_point_2 - min_point_2
    scale_dist = np.linalg.norm(scale_1 - scale_2)
    return scale_dist

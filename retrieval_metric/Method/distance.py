#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def getChamferDistance(pcd_1, pcd_2):
    dist_1_array = np.array(pcd_1.compute_point_cloud_distance(pcd_2))
    dist_2_array = np.array(pcd_2.compute_point_cloud_distance(pcd_1))

    dist_1 = np.linalg.norm(dist_1_array)
    dist_2 = np.linalg.norm(dist_2_array)

    chamfer_distance = dist_1 * dist_1 / dist_1_array.shape[
        0] + dist_2 * dist_2 / dist_2_array.shape[0]
    return chamfer_distance

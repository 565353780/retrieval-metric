#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../scan2cad-dataset-manage")
sys.path.append("../points-shape-detect")
sys.path.append("../global-to-patch-retrieval")
sys.path.append("../conv-onet")

from retrieval_metric.Module.metric_manager import MetricManager


def demo():
    print_progress = True

    metric_manager = MetricManager()

    metric_manager.getAllMetric(print_progress)
    return True

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../scan2cad-dataset-manage")

from retrieval_metric.Module.metric_manager import MetricManager


def demo():
    metric_manager = MetricManager()

    metric_manager.getAllMetric()
    return True

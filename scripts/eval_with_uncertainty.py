# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" Evaluation routine for 3D object detection with SUN RGB-D and ScanNet.
"""
#!/usr/bin/env python

from torch.utils.tensorboard import SummaryWriter
import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.abspath("/home/yildirir/workspace/votenet/README.md")
)
ROOT_DIR = BASE_DIR

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
print(sys.path)

from initialization_utils import initialize_dataloader, initialize_model, log_string

# from models.ap_helper import parse_predictions, parse_predictions_augmented, parse_predictions_ensemble_only_entropy
from ap_helper import (
    APCalculator,
    parse_predictions_ensemble,
    parse_groundtruths,
    parse_predictions,
    parse_predictions_augmented,
    aggregate_predictions,
    parse_predictions_ensemble_only_entropy,
)
from uncertainty_utils import (
    map_zero_one,
    box_size_uncertainty,
    semantic_cls_uncertainty,
    objectness_uncertainty,
    center_uncertainty,
    apply_softmax,
    compute_objectness_accuracy,
    compute_iou_masks,
    compute_iou_masks_with_classification,
)
from dump_helper import dump_results_for_sanity_check, dump_results
import pc_util


def evaluate_one_epoch(FLAGS):
    global best_map
    initialize_dataloader(FLAGS)

    net, criterion, optimizer, bnm_scheduler = initialize_model(FLAGS)

    for T in FLAGS.TEST_DATALOADERS:
        # Thresholds for AP calculation during evaluation
        CONFIG_DICT = {
            "remove_empty_box": False,
            "use_3d_nms": True,
            "nms_iou": 0.25,
            "use_old_type_nms": False,
            "cls_nms": True,
            "per_class_proposal": True,
            "conf_thresh": 0.05,
            "dataset_config": FLAGS.DATASET_CONFIG,
        }

        stat_dict = {}  # collect statistics
        ap_calculator = APCalculator(
            ap_iou_thresh=FLAGS.AP_IOU_THRESH,
            class2type_map=FLAGS.DATASET_CONFIG.class2type,
        )
        net.eval()  # set model to eval mode (for bn and dp)
        for batch_idx, batch_data_label in enumerate(T):
            if batch_idx % 10 == 0:
                print("Eval batch: %d" % (batch_idx))
            for key in batch_data_label:
                if key != "name":
                    batch_data_label[key] = batch_data_label[key].to(FLAGS.DEVICE)

            # Forward pass

            inputs = {
                "point_clouds": batch_data_label["point_clouds"],
                "name": batch_data_label["name"],
            }
            # inputs = {'point_clouds': batch_data_label['point_clouds']}
            with torch.no_grad():
                end_points = net(inputs)

            # Compute loss8
            for key in batch_data_label:
                assert key not in end_points
                end_points[key] = batch_data_label[key]
            loss, end_points = criterion(end_points, FLAGS.DATASET_CONFIG)

            # Accumulate statistics and prin t out
            # t ot
            for key in end_points:
                if "loss" in key or "acc" in key or "ratio" in key:
                    if key not in stat_dict:
                        stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

            # Dump evaluation results for visualization
            # if FLAGS.DUMP_RESULTS and batch_idx == 0 and EPOCH_CNT % 10 == 0:
            # FLAGS.MODEL.DUMP_RESULTS(end_points, FLAGS.DUMP_DIR, FLAGS.DATASET_CONFIG)
            if FLAGS.NUM_VAL_BATCHES != -1:
                if batch_idx > FLAGS.NUM_VAL_BATCHES:
                    best_map = -1
                    break
        # Log statistics

        # FLAGS.TEST_VISUALIZER.log_scalars(
        #     {key: stat_dict[key] / float(batch_idx + 1) for key in stat_dict},
        #     (EPOCH_CNT + 1) * len(FLAGS.TRAIN_DATALOADER) * FLAGS.BATCH_SIZE,
        # )

        for key in sorted(stat_dict.keys()):
            log_string(
                FLAGS.LOGGER,
                "eval mean %s: %f" % (key, stat_dict[key] / (float(batch_idx + 1))),
            )

        # Evaluate average precision
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string(FLAGS.LOGGER, "eval %s: %f" % (key, metrics_dict[key]))

        # return mean_loss, metrics_dict["mAP"]


def evaluate_with_mc_dropout(FLAGS):
    stat_dict = {}
    methods = [
        "Native",
        "objectness",
        "classification",  # TODO: add box size
    ]
    # methods = ["Native"]
    # methods = ["obj_and_cls","Native"]

    initialize_dataloader(FLAGS)
    FLAGS.CONFIG_DICT = {
        "remove_empty_box": True,
        "use_3d_nms": True,
        "nms_iou": 0.25,
        "use_old_type_nms": False,
        "cls_nms": True,
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": FLAGS.DATASET_CONFIG,
    }

    net, criterion, optimizer, bnm_scheduler = initialize_model(FLAGS)

    for T in FLAGS.TEST_DATALOADERS:
        met_dict = {}
        unc_dict = {}
        for m in methods:
            met_dict[m] = [
                APCalculator(iou_thresh, FLAGS.DATASET_CONFIG.class2type)
                for iou_thresh in FLAGS.AP_IOU_THRESHOLDS
            ]
            unc_dict[m] = [0, 0]
        net.eval()
        net.enable_dropouts()
        for batch_idx, batch_data_label in enumerate(T):
            if batch_idx % 10 == 0:
                print("Eval batch: %d" % (batch_idx))
            if batch_idx < 2:
                continue
            if batch_idx == FLAGS.NUM_VAL_BATCHES:

                break
            for key in batch_data_label:
                if key != "name":
                    batch_data_label[key] = batch_data_label[key].to(FLAGS.DEVICE)

            # Forward pass
            inputs = {
                "point_clouds": batch_data_label["point_clouds"],
                "name": batch_data_label["name"],
            }
            loss = np.zeros([FLAGS.NUM_SAMPLES])
            with torch.no_grad():
                mc_samples = [net(inputs) for i in range(FLAGS.NUM_SAMPLES)]

            for idx, end_points in enumerate(mc_samples):
                for key in batch_data_label:
                    assert key not in end_points
                    end_points[key] = batch_data_label[key]
                local_loss, end_points = criterion(end_points, FLAGS.DATASET_CONFIG)

            #     # center_uncertainty(mc_samples)
            #     #This guy has len(methods) elements
            import copy

            # print("Predictions parsing")
            batch_pred_map_cls = [
                parse_predictions_ensemble(
                    copy.deepcopy(mc_samples),
                    FLAGS.CONFIG_DICT,
                    m,
                    False,  # FLAGS.EXPECTED_ENT
                )
                for m in methods
            ]
            # print("Predictions computed")
            bsize = FLAGS.BATCH_SIZE
            org_batch_gt_map_cls = parse_groundtruths(end_points, FLAGS.CONFIG_DICT)
            # print(met_dict)
            for idx, m in enumerate(methods):
                for ap_calculator in met_dict[m]:
                    ap_calculator.step(batch_pred_map_cls[idx], org_batch_gt_map_cls)

        for idx, m in enumerate(methods):
            print("|", m, "|", "| ")
            for ap_calculator in met_dict[m]:
                print("|", "iou_thresh | %f  " % (ap_calculator.ap_iou_thresh), " | ")
                metrics_dict = ap_calculator.compute_metrics()
                for key in metrics_dict:
                    if key == "mAP" or key == "AR":
                        log_string(
                            FLAGS.LOGGER, " | %s  | %f | " % (key, metrics_dict[key])
                        )


if __name__ == "__main__":
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=1)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("C", args.config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    FLAGS = mod.C
    FLAGS.NUM_SAMPLES = args.num_samples
    FLAGS.NUM_RUNS = args.num_runs
    evaluate_one_epoch(FLAGS)
    # for i in range(FLAGS.NUM_RUNS):
    evaluate_with_mc_dropout(FLAGS)

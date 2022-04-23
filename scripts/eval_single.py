import os
import sys
import numpy as np
from datetime import datetime
import importlib.util
import argparse
import trimesh
import pdb
import glob
from natsort import natsorted  # pip install natsort

# pytorch stuff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


BASE_DIR = os.path.dirname(
    os.path.abspath("/home/yildirir/workspace/votenet/README.md")
)

ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(BASE_DIR)

from ap_helper import flip_axis_to_depth

np.random.seed(31)
# project stuff
from dump_helper import (
    dump_results_for_sanity_check,
    dump_results_mini,
    dump_only_boxes,
    dump_only_boxes_gt,
)
from ap_helper import (
    APCalculator,
    parse_predictions,
    parse_groundtruths,
    parse_predictions_with_log_var,
    compute_batch_iou,
    flip_axis_to_camera,
)
from initialization_utils import (
    initialize_dataloader,
    initialize_model,
    log_string,
)

from box_util import get_3d_box


def evaluate_with_sampling(FLAGS):
    net, criterion, optimizer, bnm_scheduler = initialize_model(FLAGS)
    net.eval()
    # TODO: D eal with dropouts, for now they are closed
    # Used for AP calculation during evaluation
    CONFIG_DICT = {
        "remove_empty_box": True,
        "use_3d_nms": True,
        "nms_iou": 0.25,
        "use_old_type_nms": False,
        "cls_nms": True,
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": FLAGS.DATASET_CONFIG,
    }

    stat_dict = {}  # collect statistics
    FLAGS.TEST_DATALOADERS = [FLAGS.TEST_DATALOADER]
    for T in FLAGS.TEST_DATALOADERS:
        ap_calculator = APCalculator(
            ap_iou_thresh=FLAGS.AP_IOU_THRESH,
            class2type_map=FLAGS.DATASET_CONFIG.class2type,
        )
        total_iou = 0
        total_num_boxes = 0
        total_sigma = 0
        bins = T.dataset.bin_thresholds
        res_bins = {}
        for b in bins:
            res_bins[b] = [0, 0, 0]
        for batch_idx, batch_data_label in enumerate(T):
            if batch_idx % 10 == 0:
                print("Eval batch: %d" % (batch_idx))
            for key in batch_data_label:
                if key != "name":
                    batch_data_label[key] = batch_data_label[key].to(FLAGS.DEVICE)

            # Forward pass
            inputs = {
                "point_clouds": batch_data_label["point_clouds"],
                "center_label": batch_data_label["center_label"],
                "name": batch_data_label["name"],
            }
            # print(inputs)
            # for idx, pc in enumerate(batch_data_label["point_clouds"]):
            #     filename = os.path.join(
            #         FLAGS.DUMP_DIR, "{}.ply".format(batch_data_label["name"][idx])
            #     )
            for idx, name in enumerate(inputs["name"]):
                trimesh.points.PointCloud(
                    # flip_axis_to_depth(
                    batch_data_label["point_clouds"][idx, :, :3]
                    .cpu()
                    .numpy()
                    # )
                ).export("{}.ply".format(batch_data_label["name"][idx]))

            with torch.no_grad():
                end_points = net(inputs)
            for key in batch_data_label:
                end_points[key] = batch_data_label[key]

            # end_points["box_label_mask"] = (
            #     end_points["box_label_mask"] * (end_points["class_labels"] == 4).cuda()
            # )
            prev = 0.3
            b = 1.0
            end_points["box_label_mask"] = (
                (end_points["score_labels"] > prev) & (end_points["score_labels"] <= b)
            ).cuda() * end_points["box_label_mask"]
            box_label_mask_vanilla = end_points["box_label_mask"].float().clone()
            iou, num_boxes = compute_batch_iou(end_points, FLAGS)
            total_iou += iou
            total_num_boxes += num_boxes
            total_sigma += (
                torch.exp(end_points["log_vars"][end_points["box_label_mask"].bool()])
                ** 0.5
            ).sum()

            # for idx, b in enumerate(bins):
            #     end_points["box_label_mask"] = (
            #         (end_points["score_labels"] > prev)
            #         & (end_points["score_labels"] <= b)
            #     ).cuda().float() * box_label_mask_vanilla
            #     # print(end_points["box_label_mask"])
            #     iou, num_boxes = compute_batch_iou(end_points, FLAGS)
            #     res_bins[b][0] += iou
            #     res_bins[b][1] += num_boxes
            #     res_bins[b][2] += (
            #         torch.exp(
            #             end_points["log_vars"][end_points["box_label_mask"].bool()]
            #         )
            #         ** 0.5
            #     ).sum()
            #     prev = b
        #     # Compute loss8
        #     for key in batch_data_label:
        #         assert key not in end_points
        #         end_points[key] = batch_data_label[key]
        #     loss, end_points = criterion(end_points, FLAGS.DATASET_CONFIG)

        #     # Accumulate statistics and prin t out
        #     # t ot
        #     for key in end_points:
        #         if "loss" in key or "acc" in key or "ratio" in key:
        #             if key not in stat_dict:
        #                 stat_dict[key] = 0
        #             stat_dict[key] += end_points[key].item()
        #     # batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        #     batch_pred_map_cls, selected_raw_boxes = parse_predictions_with_log_var(
        #         end_points, CONFIG_DICT, sampling=FLAGS.SAMPLING
        #     )
        #     # print(batch_pred_map_cls)
        #     batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        #     ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        #     # Dump evaluation results for visualization
        #     # if FLAGS.DUMP_RESULTS and batch_idx == 0 and EPOCH_CNT % 10 == 0:
        #     # FLAGS.MODEL.DUMP_RESULTS(end_points, FLAGS.DUMP_DIR, FLAGS.DATASET_CONFIG)
        # # Log statistics

        # # FLAGS.TEST_VISUALIZER.log_scalars(
        # #     {key: stat_dict[key] / float(batch_idx + 1) for key in stat_dict},
        # #     (EPOCH_CNT + 1) * len(FLAGS.TRAIN_DATALOADER) * FLAGS.BATCH_SIZE,
        # # )
        # for key in sorted(stat_dict.keys()):
        #     log_string(
        #         FLAGS.LOGGER,
        #         "eval mean %s: %f" % (key, stat_dict[key] / (float(batch_idx + 1))),
        #     )

        # # Evaluate average precision
        # metrics_dict = ap_calculator.compute_metrics()
        # for key in metrics_dict:
        #     log_string(FLAGS.LOGGER, "eval %s: %f" % (key, metrics_dict[key]))
        # dump_results_mini(
        #     end_points,
        #     config=FLAGS.DATASET_CONFIG,
        #     dump_dir=FLAGS.DUMP_DIR + str(T.dataset.thresh),
        # )
        print(
            total_iou / total_num_boxes, total_sigma / total_num_boxes, total_num_boxes
        )
        # print(res_bins)
        # res_bins["all"] = total_iou / total_num_boxes
        # for b in bins:

        #     print(
        #         "Average IOU bin ",
        #         b,
        #         res_bins[b][0] / res_bins[b][1],
        #         "sigma",
        #         res_bins[b][2] / res_bins[b][1],
        #         res_bins[b][1],
        #     )
        # dump_only_boxes(selected_raw_boxes, FLAGS.DUMP_DIR)
        # dump_only_boxes_gt(batch_gt_map_cls, FLAGS.DUMP_DIR)
        return res_bins


def save_datas(FLAGS):
    net, criterion, optimizer, bnm_scheduler = initialize_model(FLAGS)
    net.eval()
    CONFIG_DICT = {
        "remove_empty_box": False,
        "use_3d_nms": True,
        "nms_iou": 0.25,
        "use_old_type_nms": False,
        "cls_nms": True,
        "per_class_proposal": False,
        "conf_thresh": 0.05,
        "dataset_config": FLAGS.DATASET_CONFIG,
    }

    stat_dict = {}  # collect statistics
    FLAGS.TEST_DATALOADERS = [FLAGS.TEST_DATALOADER]
    for T in FLAGS.TEST_DATALOADERS:
        ap_calculator = APCalculator(
            ap_iou_thresh=FLAGS.AP_IOU_THRESH,
            class2type_map=FLAGS.DATASET_CONFIG.class2type,
        )

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
                "center_label": batch_data_label["center_label"],
            }
            with torch.no_grad():
                end_points = net(inputs)
            bsize = len(batch_data_label["name"])
            for b in range(bsize):
                pred_sizes = (
                    end_points["size_preds"][b]
                    + torch.from_numpy(FLAGS.DATASET_CONFIG.type_mean_size["chair"])
                    .cuda()
                    .float()
                )
                log_vars = torch.exp(end_points["log_vars"][b]) ** 0.5
                name = batch_data_label["name"][b]
                gt_centers = batch_data_label["center_label"][b][
                    batch_data_label["box_label_mask"][b].bool()
                ]
                gt_sizes = (
                    torch.from_numpy(FLAGS.DATASET_CONFIG.type_mean_size["chair"])
                    .cuda()
                    .float()
                    + batch_data_label["size_residual_label"][b].float()
                )[batch_data_label["box_label_mask"][b].bool()]
                # gt_centers = torch.from_numpy(flip_axis_to_camera(gt_sizes.cpu()))

                # print(gt_sizes, gt_centers)
                gt_sizes = torch.from_numpy(flip_axis_to_depth(gt_sizes.cpu()))
                pred_sizes = torch.from_numpy(flip_axis_to_depth(pred_sizes.cpu()))
                # print(gt_sizes, gt_centers)

                boxes = []
                scene = trimesh.scene.Scene()
                # scene.add_geometry(

                #     trimesh.points.PointCloud(
                #         batch_data_label["point_clouds"][b, :, :3].cpu().numpy()
                #     )
                # )
                for idx in range(len(gt_centers)):
                    corners = get_3d_box(
                        gt_sizes[idx].cpu().numpy(), 0, gt_centers[idx].cpu().numpy()
                    )
                    # print(corners)
                    scene.add_geometry(trimesh.points.PointCloud(corners).convex_hull)
                # save to ply file
                #
                [
                    a.export(name + "_gt_" + str(idx) + ".ply")
                    for idx, a in enumerate(scene.dump())
                ]

                trimesh.points.PointCloud(
                    batch_data_label["point_clouds"][b, :, :3].cpu().numpy()
                ).export("{}.ply".format(name))

                # pdb.set_trace()
                boxes = []
                scene = trimesh.scene.Scene()
                for idx in range(len(gt_centers)):
                    corners = get_3d_box(
                        pred_sizes[idx].cpu().numpy(), 0, gt_centers[idx].cpu().numpy()
                    )
                    scene.add_geometry(trimesh.points.PointCloud(corners).convex_hull)
                # save to ply file
                [
                    a.export(
                        name
                        + "_pred_"
                        + str(idx)
                        + "_{}.ply".format("%.2f" % log_vars[idx])
                    )
                    for idx, a in enumerate(scene.dump())
                ]

            # h
            # inputs = {'point_clouds': batch_data_label['point_clouds']}
            #
            # # Compute loss8
            # for key in batch_data_label:
            #     assert key not in end_points
            #     end_points[key] = batch_data_label[key]
            # loss, end_points = criterion(end_points, FLAGS.DATASET_CONFIG)

            # # Accumulate Statistics And Prin T out
            # # t ot
            # for key in end_points:
            #     if "loss" in key or "acc" in key or "ratio" in key:
            #         if key not in stat_dict:
            #             stat_dict[key] = 0
            #         stat_dict[key] += end_points[key].item()

            # # batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
            # batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
            # ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

            # Dump evaluation results for visualization
            # if FLAGS.DUMP_RESULTS and batch_idx == 0 and EPOCH_CNT % 10 == 0:
            # FLAGS.MODEL.DUMP_RESULTS(end_points, FLAGS.DUMP_DIR, FLAGS.DATASET_CONFIG)
        # Log statistics

        # FLAGS.TEST_VISUALIZER.log_scalars(
        #     {key: stat_dict[key] / float(batch_idx + 1) for key in stat_dict},
        #     (EPOCH_CNT + 1) * len(FLAGS.TRAIN_DATALOADER) * FLAGS.BATCH_SIZE,
        # )

        # for key in sorted(stat_dict.keys()):
        #     log_string(
        #         FLAGS.LOGGER,
        #         "eval mean %s: %f" % (key, stat_dict[key] / (float(batch_idx + 1))),
        #     )

        # # Evaluate average precision
        # metrics_dict = ap_calculator.compute_metrics()
        # for key in metrics_dict:
        #     log_string(FLAGS.LOGGER, "eval %s: %f" % (key, metrics_dict[key]))
        # dump_results_mini(
        #     end_points,
        #     config=FLAGS.DATASET_CONFIG,
        #     dump_dir=FLAGS.DUMP_DIR + str(T.dataset.thresh),
        # )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path")
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--multi_cp")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("C", args.config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    FLAGS = mod.C
    if args.multi_cp:
        files = natsorted(glob.glob(args.multi_cp + "/*.tar"))
        # print(files)
        for f in files:
            spec = importlib.util.spec_from_file_location("C", args.config_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            FLAGS = mod.C

            FLAGS.CHECKPOINT_PATH = f
            # FLAGS.SAMPLING = args.sampling
            initialize_dataloader(FLAGS)
            # save_datas(FLAGS)
            evaluate_with_sampling(FLAGS)

    else:
        initialize_dataloader(FLAGS)
        # save_datas(FLAGS)
        evaluate_with_sampling(FLAGS)

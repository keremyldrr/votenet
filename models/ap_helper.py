# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
from eval_det import eval_det_cls, eval_det_multiprocessing, eval_det, eval_det_iou
from eval_det import get_iou_obb
from nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from box_util import get_3d_box
from nn_distance import nn_distance, huber_loss

from uncertainty_utils import accumulate_mc_samples, accumulate_scores
import pc_util

sys.path.append(os.path.join(ROOT_DIR, "sunrgbd"))
from sunrgbd_utils import extract_pc_in_box3d
from box_util import box3d_iou
from nn_distance import nn_distance_iou


def flip_axis_to_camera(pc):
    """Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2


def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2


def softmax(x):
    """Numpy function for softmax"""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def end_pts_to_bb(
    end_points,
    config_dict,
):
    pred_center = end_points["center"]  # B,num_proposal,3
    pred_variances = torch.exp(end_points["log_vars"]) ** 0.5

    pred_heading_class = torch.argmax(
        end_points["heading_scores"], -1
    )  # B,num_proposal
    pred_heading_residual = torch.gather(
        end_points["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)
    )  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points["size_scores"], -1)  # B,num_proposal
    pred_size_residual = torch.gather(
        end_points["size_residuals"],
        2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
    )  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points["sem_cls_scores"], -1)  # B,num_proposal
    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))

    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera
            # pred_box_sizes[i, j] = (
            #     np.linalg.norm(
            #         corners_3d_upright_camera[0] - corners_3d_upright_camera[1]
            #     )
            #     * np.linalg.norm(
            #         corners_3d_upright_camera[2] - corners_3d_upright_camera[1]
            #     )
            #     * np.linalg.norm(
            #         corners_3d_upright_camera[4] - corners_3d_upright_camera[1]
            #     )
            # )

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))
    end_points["raw_pred_boxes"] = pred_corners_3d_upright_camera
    if config_dict["remove_empty_box"]:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points["point_clouds"].cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):

                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                if not noflip:
                    box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------

    obj_logits = end_points["objectness_scores"].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    if not config_dict["use_3d_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_2d_faster(
                boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i, j
                ]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
        return pred_corners_3d_upright_camera, pred_mask, pred_center, end_points


def parse_predictions_with_log_var(end_points, config_dict):
    """Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    pred_corners_3d_upright_camera, pred_mask, pred_center, end_points = end_pts_to_bb(
        end_points, config_dict
    )
    print(end_points["raw_pred_boxes"].shape)
    pred_variances = torch.exp(end_points["log_vars"]) ** 0.5
    obj_logits = end_points["objectness_scores"].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  #
    sem_cls_probs = softmax(
        end_points["sem_cls_scores"].detach().cpu().numpy()
    )  # B,num_proposal,10

    bsize = pred_center.shape[0]
    pred_sem_cls = torch.argmax(end_points["sem_cls_scores"], -1)  # B,num_proposal
    batch_pred_map_cls = (
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    # final_mask = np.zeros_like(pred_mask)
    selected_raw_boxes = []
    for i in range(bsize):

        if config_dict["per_class_proposal"]:
            cur_list = []
            for ii in range(config_dict["dataset_config"].num_class):
                cur_list += [
                    (
                        ii,
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, ii] * obj_prob[i, j],
                    )
                    for j in range(pred_center.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                    )
                    for j in range(pred_center.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
            selected_raw_boxes.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_variances[i, j],
                        pred_corners_3d_upright_camera[i, j],
                    )
                    for j in range(pred_center.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
            # for j in range(pred_center.shape[1]):
            #     if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']:

    # import pdb

    # pdb.set_trace()
    end_points["batch_pred_map_cls"] = batch_pred_map_cls
    # end_points["final_masks"] = final_mask
    return batch_pred_map_cls, selected_raw_boxes


def parse_predictions(end_points, config_dict, noflip=False):
    """Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    pred_center = end_points["center"]  # B,num_proposal,3
    pred_heading_class = torch.argmax(
        end_points["heading_scores"], -1
    )  # B,num_proposal
    pred_heading_residual = torch.gather(
        end_points["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)
    )  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points["size_scores"], -1)  # B,num_proposal
    pred_size_residual = torch.gather(
        end_points["size_residuals"],
        2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
    )  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points["sem_cls_scores"], -1)  # B,num_proposal
    sem_cls_probs = softmax(
        end_points["sem_cls_scores"].detach().cpu().numpy()
    )  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal
    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    if noflip:
        pred_center_upright_camera = pred_center.detach().cpu().numpy()
    else:
        pred_center_upright_camera = flip_axis_to_camera(
            pred_center.detach().cpu().numpy()
        )
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_box_sizes = np.zeros((bsize, num_proposal))

    if noflip:
        pred_center_upright_camera = pred_center.detach().cpu().numpy()
    else:
        pred_center_upright_camera = flip_axis_to_camera(
            pred_center.detach().cpu().numpy()
        )
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera
            pred_box_sizes[i, j] = (
                np.linalg.norm(
                    corners_3d_upright_camera[0] - corners_3d_upright_camera[1]
                )
                * np.linalg.norm(
                    corners_3d_upright_camera[2] - corners_3d_upright_camera[1]
                )
                * np.linalg.norm(
                    corners_3d_upright_camera[4] - corners_3d_upright_camera[1]
                )
            )

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))
    end_points["pred_box_sizes"] = torch.Tensor(pred_box_sizes)
    end_points["raw_pred_boxes"] = pred_corners_3d_upright_camera
    if config_dict["remove_empty_box"]:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points["point_clouds"].cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):

                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                if not noflip:
                    box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------

    obj_logits = end_points["objectness_scores"].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    if not config_dict["use_3d_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_2d_faster(
                boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i, j
                ]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = (
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    # final_mask = np.zeros_like(pred_mask)
    for i in range(bsize):
        if config_dict["per_class_proposal"]:
            cur_list = []
            for ii in range(config_dict["dataset_config"].num_class):
                cur_list += [
                    (
                        ii,
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, ii] * obj_prob[i, j],
                    )
                    for j in range(pred_center.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                    )
                    for j in range(pred_center.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
            # for j in range(pred_center.shape[1]):
            #     if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']:
            #         final_mask[i,j] = 1

    end_points["batch_pred_map_cls"] = batch_pred_map_cls
    # end_points["final_masks"] = final_mask
    return batch_pred_map_cls


def parse_predictions_with_objectness_prob(end_points, config_dict):
    """Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    pred_center = end_points["center"]  # B,num_proposal,3
    pred_heading_class = torch.argmax(
        end_points["heading_scores"], -1
    )  # B,num_proposal
    pred_heading_residual = torch.gather(
        end_points["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)
    )  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points["size_scores"], -1)  # B,num_proposal
    pred_size_residual = torch.gather(
        end_points["size_residuals"],
        2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
    )  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points["sem_cls_scores"], -1)  # B,num_proposal
    sem_cls_probs = softmax(
        end_points["sem_cls_scores"].detach().cpu().numpy()
    )  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal
    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_box_sizes = np.zeros((bsize, num_proposal))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera
            pred_box_sizes[i, j] = (
                np.linalg.norm(
                    corners_3d_upright_camera[0] - corners_3d_upright_camera[1]
                )
                * np.linalg.norm(
                    corners_3d_upright_camera[2] - corners_3d_upright_camera[1]
                )
                * np.linalg.norm(
                    corners_3d_upright_camera[4] - corners_3d_upright_camera[1]
                )
            )

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))
    end_points["pred_box_sizes"] = torch.Tensor(pred_box_sizes)
    end_points["raw_pred_boxes"] = pred_corners_3d_upright_camera
    if config_dict["remove_empty_box"]:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points["point_clouds"].cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------

    obj_logits = end_points["objectness_scores"].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)

    # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = (
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    final_mask = np.zeros_like(nonempty_box_mask)
    for i in range(bsize):
        for j in range(pred_center.shape[1]):
            if obj_prob[i, j] > config_dict["conf_thresh"]:
                final_mask[i, j] = 1

    end_points["batch_pred_map_cls"] = batch_pred_map_cls
    end_points["final_masks"] = final_mask
    return batch_pred_map_cls


def parse_predictions_with_custom_mask(end_points, config_dict):
    """Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    pred_center = end_points["center"]  # B,num_proposal,3
    pred_heading_class = torch.argmax(
        end_points["heading_scores"], -1
    )  # B,num_proposal
    pred_heading_residual = torch.gather(
        end_points["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)
    )  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points["size_scores"], -1)  # B,num_proposal
    pred_size_residual = torch.gather(
        end_points["size_residuals"],
        2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
    )  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points["sem_cls_scores"], -1)  # B,num_proposal
    sem_cls_probs = softmax(
        end_points["sem_cls_scores"].detach().cpu().numpy()
    )  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal
    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_box_sizes = np.zeros((bsize, num_proposal))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera
            pred_box_sizes[i, j] = (
                np.linalg.norm(
                    corners_3d_upright_camera[0] - corners_3d_upright_camera[1]
                )
                * np.linalg.norm(
                    corners_3d_upright_camera[2] - corners_3d_upright_camera[1]
                )
                * np.linalg.norm(
                    corners_3d_upright_camera[4] - corners_3d_upright_camera[1]
                )
            )

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))
    end_points["pred_box_sizes"] = torch.Tensor(pred_box_sizes)
    end_points["raw_pred_boxes"] = pred_corners_3d_upright_camera
    # if config_dict['remove_empty_box']:
    #     # -------------------------------------
    #     # Remove predicted boxes without any point within them..
    #     batch_pc = end_points['point_clouds'].cpu().numpy()[:, :, 0:3]  # B,N,3
    #     for i in range(bsize):
    #         pc = batch_pc[i, :, :]  # (N,3)
    #         for j in range(K):
    #             box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
    #             box3d = flip_axis_to_depth(box3d)
    #             pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
    #             if len(pc_in_box) < 5:
    #                 nonempty_box_mask[i, j] = 0
    #     # -------------------------------------

    obj_logits = end_points["objectness_scores"].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    custom_mask = end_points["custom_mask"]
    if config_dict["use_3d_nms"] and config_dict["cls_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            nonempty_box_mask[i] = custom_mask
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i, j
                ]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
    # pred_mask = np.logical_and(pred_mask[0],)
    batch_pred_map_cls = (
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    # final_mask = np.zeros_like(pred_mask)
    for i in range(bsize):
        batch_pred_map_cls.append(
            [
                (
                    pred_sem_cls[i, j].item(),
                    pred_corners_3d_upright_camera[i, j],
                    obj_prob[i, j],
                )
                for j in range(pred_center.shape[1])
                if pred_mask[i, j] == 1
            ]
        )
        # if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']:
        #     final_mask[i,j] = 1

    end_points["batch_pred_map_cls"] = batch_pred_map_cls
    # end_points["final_masks"] = final_mask
    return batch_pred_map_cls


def parse_predictions_ensemble(
    mc_samples, config_dict, extension=None, expected_ent=None
):
    """Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: list of dicts, results of MC sampling
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    end_points = accumulate_mc_samples(mc_samples, classification=expected_ent)
    pred_center = end_points["center"]  # B,num_proposal,3
    pred_heading_class = torch.argmax(
        end_points["heading_scores"], -1
    )  # B,num_proposal
    pred_heading_residual = torch.gather(
        end_points["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)
    )  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points["size_scores"], -1)  # B,num_proposal
    pred_size_residual = torch.gather(
        end_points["size_residuals"],
        2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
    )  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points["sem_cls_scores"], -1)  # B,num_proposal
    sem_cls_probs = softmax(
        end_points["sem_cls_scores"].detach().cpu().numpy()
    )  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal
    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_box_sizes = np.zeros((bsize, num_proposal))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera
            # pred_box_sizes[i, j] = np.linalg.norm(
            #     corners_3d_upright_camera[0] - corners_3d_upright_camera[1]
            # ) * np.linalg.norm(corners_3d_upright_camera[2] -
            #                    corners_3d_upright_camera[1]) * np.linalg.norm(
            #                        corners_3d_upright_camera[4] -
            #                        corners_3d_upright_camera[1])

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))
    # end_points["pred_box_sizes"] = torch.Tensor(pred_box_sizes)
    # end_points["raw_pred_boxes"] = pred_corners_3d_upright_camera
    if config_dict["remove_empty_box"]:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points["point_clouds"].cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)

                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------

    obj_logits = end_points["objectness_scores"].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    cls_entropy = end_points["semantic_cls_entropy"]
    obj_entropy = end_points["objectness_entropy"]
    extra = np.ones_like(obj_entropy)
    if extension == "objectness":
        extra = 1 - obj_entropy
    elif extension == "classification":
        extra = 1 - cls_entropy
    elif extension == "obj_and_cls":
        extra = (1 - obj_entropy) * (1 - cls_entropy)

    obj_prob = obj_prob * extra
    # box_size_entropy = end_points["box_size_entropy"]
    if not config_dict["use_3d_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_2d_faster(
                boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i, j
                ]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = (
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    # final_mask = np.zeros_like(pred_mask)

    # print(extension)
    for i in range(bsize):

        if config_dict["per_class_proposal"]:
            cur_list = []
            for ii in range(config_dict["dataset_config"].num_class):
                cur_list += [
                    (
                        ii,
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, ii] * obj_prob[i, j],
                    )
                    for j in range(pred_center.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            batch_pred_map_cls.append(cur_list)
            # print("Accepted" ,len(cur_list))
        else:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                    )
                    for j in range(pred_center.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
            # for j in range(pred_center.shape[1]):
            #     if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']:
            #         final_mask[i,j] = 1
    # print("*******************")
    end_points["batch_pred_map_cls"] = batch_pred_map_cls
    # print(extension,sum([len(a) for a in batch_pred_map_cls]),obj_prob)

    # end_points["final_masks"] = final_mask
    return batch_pred_map_cls


def parse_predictions_ensemble_only_entropy(mc_samples, config_dict, extension=None):
    """Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: list of dicts, results of MC sampling
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    # end_points = accumulate_mc_samples(mc_samples)

    # obj_logits = end_points['objectness_scores'].detach().cpu().numpy()
    # obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    # cls_entropy = end_points["semantic_cls_entropy"]
    # obj_entropy = end_points["objectness_entropy"]
    # extra = np.ones_like(obj_entropy)
    # if extension == "objectness":
    #     extra = 1 - obj_entropy
    # elif extension == "classification":
    #     extra = 1 - cls_entropy
    # elif extension == "obj_and_cls":
    #     extra = (1 - obj_entropy )*(1 -  cls_entropy)

    # obj_prob =obj_prob *  extra

    end_points = accumulate_mc_samples(mc_samples)
    pred_center = end_points["center"]  # B,num_proposal,3
    pred_heading_class = torch.argmax(
        end_points["heading_scores"], -1
    )  # B,num_proposal
    pred_heading_residual = torch.gather(
        end_points["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)
    )  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points["size_scores"], -1)  # B,num_proposal
    pred_size_residual = torch.gather(
        end_points["size_residuals"],
        2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
    )  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points["sem_cls_scores"], -1)  # B,num_proposal
    sem_cls_probs = softmax(
        end_points["sem_cls_scores"].detach().cpu().numpy()
    )  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal
    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_box_sizes = np.zeros((bsize, num_proposal))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera
            pred_box_sizes[i, j] = (
                np.linalg.norm(
                    corners_3d_upright_camera[0] - corners_3d_upright_camera[1]
                )
                * np.linalg.norm(
                    corners_3d_upright_camera[2] - corners_3d_upright_camera[1]
                )
                * np.linalg.norm(
                    corners_3d_upright_camera[4] - corners_3d_upright_camera[1]
                )
            )

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))
    end_points["pred_box_sizes"] = torch.Tensor(pred_box_sizes)
    end_points["raw_pred_boxes"] = pred_corners_3d_upright_camera
    if config_dict["remove_empty_box"]:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points["point_clouds"].cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)

                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------

    obj_logits = end_points["objectness_scores"].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    cls_entropy = end_points["semantic_cls_entropy"]
    obj_entropy = end_points["objectness_entropy"]
    extra = np.ones_like(obj_entropy)
    if extension == "objectness":
        extra = 1 - obj_entropy
    elif extension == "classification":
        extra = 1 - cls_entropy
    elif extension == "obj_and_cls":
        extra = (1 - obj_entropy) * (1 - cls_entropy)

    obj_prob = obj_prob * extra
    # box_size_entropy = end_points["box_size_entropy"]
    if not config_dict["use_3d_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_2d_faster(
                boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i, j
                ]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points["pred_mask"] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = (
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    # final_mask = np.zeros_like(pred_mask)

    # print(extension)
    print("Batch size ", bsize)
    for i in range(bsize):

        if bsize > 1:

            batch_pred_map_cls.append(
                [
                    (
                        (obj_entropy + cls_entropy)[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                    )
                    for j in range(pred_center.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
        else:

            batch_pred_map_cls.append(
                [
                    (
                        (obj_entropy + cls_entropy)[j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                    )
                    for j in range(pred_center.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )

    end_points["batch_pred_map_cls"] = batch_pred_map_cls
    # end_points["final_masks"] = final_mask

    # box_size_entropy = end_points["box_size_entropy"]
    # ---------- NMS output: pred_mask in (B,K) -----------

    obj_and_cls_uncertainties = np.array(
        [np.sum([(a[0]) for a in batch_pred_map_cls[i]]) for i in range(bsize)]
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    # final_mask = np.zeros_like(pred_mask)

    # print(extension)

    # end_points["final_masks"] = final_mask
    return obj_and_cls_uncertainties


def make_box_and_unrotate(end_points, config_dict, rot):

    pred_center = end_points["center"].cpu().numpy()  # B,num_proposal,3
    pred_heading_class = torch.argmax(
        end_points["heading_scores"], -1
    )  # B,num_proposal
    pred_heading_residual = torch.gather(
        end_points["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)
    )  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points["size_scores"], -1)  # B,num_proposal
    pred_size_residual = torch.gather(
        end_points["size_residuals"],
        2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
    )  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)

    bsize = pred_center.shape[0]
    num_proposal = pred_center.shape[1]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    box_sizes = np.zeros((bsize, num_proposal))

    # print("*********************")

    # print(pred_center[0,0])
    # print("*********************")
    # pred_center_upright_camera = flip_axis_to_camera(pred_center)
    pred_center_upright_camera = pred_center
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict["dataset_config"].class2angle(
                pred_heading_class[i, j].detach().cpu().numpy(),
                pred_heading_residual[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(pred_size_class[i, j].detach().cpu().numpy()),
                pred_size_residual[i, j].detach().cpu().numpy(),
            )
            box_sizes[i, j] = box_size[0] * box_size[1] * box_size[2]
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, pred_center_upright_camera[i, j, :]
            )
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    for i in range(bsize):
        for j in range(num_proposal):
            # rot = np.eye(3)
            pred_center[i, j], _ = pc_util.rotate_point_cloud(
                pred_center_upright_camera[i, j], rotation_matrix=rot.T
            )  # do rotation to all points here
            pred_center[i, j] = flip_axis_to_camera(pred_center[i, j])
            pred_corners_3d_upright_camera[i, j], _ = pc_util.rotate_point_cloud(
                pred_corners_3d_upright_camera[i, j], rotation_matrix=rot.T
            )
            pred_corners_3d_upright_camera[i, j] = flip_axis_to_camera(
                pred_corners_3d_upright_camera[i, j]
            )
            # boxes = pc_util.point_cloud_to_bbox(pred_corners_3d_upright_camera[i, j])
            # if (rot == np.eye(3)).sum() != 0:
            #     print("Box center ",boxes[:3])
            #     print("Pred center ",pred_center[i, j])

    return pred_corners_3d_upright_camera, pred_center, box_sizes
    # return pred_center


def parse_predictions_augmented(
    mc_samples, config_dict, extension=None, rotations=None
):
    """Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: list of dicts, results of MC sampling
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    # inds = np.array([0,1,3,2])
    rotations = np.array(rotations)
    pred_corners_3d_upright_camera, pred_center, box_sizes = map(
        list,
        zip(
            *[
                make_box_and_unrotate(mc_samples[idx], config_dict, rot)
                for idx, rot in enumerate(rotations)
            ]
        ),
    )
    # K = pred_corners_3d_upright_camera[0].shape[1]  # K==num_proposal
    # bsize = 1# pred_corners_3d_upright_camera[0].shape[0]
    # numbox = 10
    # for idx,pred_boxes in enumerate(pred_corners_3d_upright_camera):
    #     boxes_to_be_dumped = []
    #     for b in range(bsize):

    #         for j in range(numbox):
    #             pcd_to_box = pc_util.point_cloud_to_bbox(pred_boxes[b,j])
    #             boxes_to_be_dumped.append(pcd_to_box)
    #     boxes_to_be_dumped = np.array(boxes_to_be_dumped)
    #     pc_util.write_bbox(boxes_to_be_dumped,str(idx) + ".ply")

    end_points = accumulate_scores(mc_samples)
    # end_points = mc_samples[3]
    pred_corners_3d_upright_camera = (
        torch.Tensor(pred_corners_3d_upright_camera).mean(dim=0).cpu().numpy()
    )
    pred_center = torch.Tensor(pred_center).mean(dim=0).cpu().numpy()
    pred_sem_cls = (
        torch.argmax(end_points["sem_cls_scores"], -1).cpu().numpy()
    )  # B,num_proposal
    sem_cls_probs = softmax(
        end_points["sem_cls_scores"].detach().cpu().numpy()
    )  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal

    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.

    K = pred_corners_3d_upright_camera.shape[1]  # K==num_proposal
    bsize = pred_corners_3d_upright_camera.shape[0]

    nonempty_box_mask = np.ones((bsize, K))
    end_points["raw_pred_boxes"] = pred_corners_3d_upright_camera
    if config_dict["remove_empty_box"]:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points["point_clouds"].cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------

    obj_logits = end_points["objectness_scores"].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    # cls_entropy = end_points["semantic_cls_entropy"]
    # obj_entropy = end_points["objectness_entropy"]
    # extra = np.ones_like(obj_entropy)
    # if extension == "objectness":
    #     extra = 1 - obj_entropy
    # elif extension == "classification":
    #     extra = 1 - cls_entropy
    # elif extension == "obj_and_cls":
    #     extra = (1 - obj_entropy )*(1 -  cls_entropy)

    # obj_prob =obj_prob *  extra

    # ---------- NMS output: pred_mask in (B,K) -----------
    # ---------- NMS input: pred_with_prob in (B,K,8) -----------
    pred_mask = np.zeros((bsize, K))
    for i in range(bsize):
        boxes_3d_with_prob = np.zeros((K, 8))
        for j in range(K):
            boxes_3d_with_prob[j, 0] = np.min(
                pred_corners_3d_upright_camera[i, j, :, 0]
            )
            boxes_3d_with_prob[j, 1] = np.min(
                pred_corners_3d_upright_camera[i, j, :, 1]
            )
            boxes_3d_with_prob[j, 2] = np.min(
                pred_corners_3d_upright_camera[i, j, :, 2]
            )
            boxes_3d_with_prob[j, 3] = np.max(
                pred_corners_3d_upright_camera[i, j, :, 0]
            )
            boxes_3d_with_prob[j, 4] = np.max(
                pred_corners_3d_upright_camera[i, j, :, 1]
            )
            boxes_3d_with_prob[j, 5] = np.max(
                pred_corners_3d_upright_camera[i, j, :, 2]
            )
            boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            boxes_3d_with_prob[j, 7] = pred_sem_cls[
                i, j
            ]  # only suppress if the two boxes are of the same class!!
        nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
        pick = nms_3d_faster_samecls(
            boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
            config_dict["nms_iou"],
            config_dict["use_old_type_nms"],
        )
        assert len(pick) > 0
        pred_mask[i, nonempty_box_inds[pick]] = 1
    end_points["pred_mask"] = pred_mask
    # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = (
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    # final_mask = np.zeros_like(pred_mask)

    # print(extension)
    for i in range(bsize):

        if config_dict["per_class_proposal"]:
            cur_list = []
            for ii in range(config_dict["dataset_config"].num_class):
                cur_list += [
                    (
                        ii,
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, ii] * obj_prob[i, j],
                    )
                    for j in range(pred_center.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            batch_pred_map_cls.append(cur_list)
            # print("Accepted" ,len(cur_list))
        else:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                    )
                    for j in range(pred_center.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
            # for j in range(pred_center.shape[1]):
            #     if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']:
            #         final_mask[i,j] = 1
    # print("*******************")
    end_points["batch_pred_map_cls"] = batch_pred_map_cls
    # end_points["final_masks"] = final_mask
    return batch_pred_map_cls


def parse_groundtruths(end_points, config_dict):
    """Parse groundtruth labels to OBB parameters.

    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    center_label = end_points["center_label"]
    heading_class_label = end_points["heading_class_label"]
    heading_residual_label = end_points["heading_residual_label"]
    size_class_label = end_points["size_class_label"]
    size_residual_label = end_points["size_residual_label"]
    box_label_mask = end_points["box_label_mask"]
    sem_cls_label = end_points["sem_cls_label"]
    bsize = center_label.shape[0]

    K2 = center_label.shape[1]  # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    gt_box_sizes = np.zeros((bsize, K2))

    gt_center_upright_camera = flip_axis_to_camera(
        center_label[:, :, 0:3].detach().cpu().numpy()
    )
    for i in range(bsize):
        for j in range(K2):
            if box_label_mask[i, j] == 0:
                continue
            heading_angle = config_dict["dataset_config"].class2angle(
                heading_class_label[i, j].detach().cpu().numpy(),
                heading_residual_label[i, j].detach().cpu().numpy(),
            )
            box_size = config_dict["dataset_config"].class2size(
                int(size_class_label[i, j].detach().cpu().numpy()),
                size_residual_label[i, j].detach().cpu().numpy(),
            )
            corners_3d_upright_camera = get_3d_box(
                box_size, heading_angle, gt_center_upright_camera[i, j, :]
            )
            gt_corners_3d_upright_camera[i, j] = corners_3d_upright_camera
            gt_box_sizes[i, j] = (
                np.linalg.norm(
                    corners_3d_upright_camera[0] - corners_3d_upright_camera[1]
                )
                * np.linalg.norm(
                    corners_3d_upright_camera[2] - corners_3d_upright_camera[1]
                )
                * np.linalg.norm(
                    corners_3d_upright_camera[4] - corners_3d_upright_camera[1]
                )
            )

        # gt_box_sizes[i] = gt_box_sizes[i,:np.argmin(gt_box_sizes[i])]

    end_points["gt_box_sizes"] = gt_box_sizes
    end_points["raw_gt_boxes"] = gt_corners_3d_upright_camera
    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append(
            [
                (sem_cls_label[i, j].item(), gt_corners_3d_upright_camera[i, j])
                for j in range(gt_corners_3d_upright_camera.shape[1])
                if box_label_mask[i, j] == 1
            ]
        )
    end_points["batch_gt_map_cls"] = batch_gt_map_cls

    return batch_gt_map_cls


def find_matching_boxes_center(proposals):
    """
    Args
    boxes1,boxes2: list of tuples -> (class,box corners, objectness_score)
    TODO:extend to batch
    """
    B = len(proposals[0])
    # B=1
    FAR_THRESHOLD = 0.6
    NEAR_THRESHOLD = 0.3
    ACCEPTANCE_THRESHOLD = 4
    print("ACCEPTANCE THRESHOLD ", ACCEPTANCE_THRESHOLD)
    final_proposals = []
    for b in range(B):
        centers1 = torch.from_numpy(
            np.array([pc_util.point_cloud_to_bbox(bx[1])[:3] for bx in proposals[0][b]])
        ).unsqueeze(0)
        centers2 = torch.from_numpy(
            np.array([pc_util.point_cloud_to_bbox(bx[1])[:3] for bx in proposals[1][b]])
        ).unsqueeze(0)
        centers3 = torch.from_numpy(
            np.array([pc_util.point_cloud_to_bbox(bx[1])[:3] for bx in proposals[2][b]])
        ).unsqueeze(0)
        centers4 = torch.from_numpy(
            np.array([pc_util.point_cloud_to_bbox(bx[1])[:3] for bx in proposals[3][b]])
        ).unsqueeze(0)

        K = centers1.shape[1]
        K2 = centers1.shape[0]
        dist12, _, _, _ = nn_distance(centers1, centers2)  # dist1: BxK, dist2: BxK2
        dist13, _, _, _ = nn_distance(centers1, centers3)  # dist1: BxK, dist2: BxK2
        dist14, _, _, _ = nn_distance(centers1, centers4)  # dist1: BxK, dist2: BxK2

        # Generate objectness label and mask
        # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
        # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
        euclidean_dist12 = torch.sqrt(dist12 + 1e-6)
        objectness_label12 = torch.zeros((1, K), dtype=torch.long).cuda()
        objectness_mask12 = torch.zeros((1, K)).cuda()
        objectness_label12[euclidean_dist12 < NEAR_THRESHOLD] = 1
        objectness_mask12[euclidean_dist12 < NEAR_THRESHOLD] = 1
        objectness_mask12[euclidean_dist12 > FAR_THRESHOLD] = 1

        euclidean_dist13 = torch.sqrt(dist13 + 1e-6)
        objectness_label13 = torch.zeros((1, K), dtype=torch.long).cuda()
        objectness_mask13 = torch.zeros((1, K)).cuda()
        objectness_label13[euclidean_dist13 < NEAR_THRESHOLD] = 1
        objectness_mask13[euclidean_dist13 < NEAR_THRESHOLD] = 1
        objectness_mask13[euclidean_dist13 > FAR_THRESHOLD] = 1

        euclidean_dist14 = torch.sqrt(dist14 + 1e-6)
        objectness_label14 = torch.zeros((1, K), dtype=torch.long).cuda()
        objectness_mask14 = torch.zeros((1, K)).cuda()
        objectness_label14[euclidean_dist14 < NEAR_THRESHOLD] = 1
        objectness_mask14[euclidean_dist14 < NEAR_THRESHOLD] = 1
        objectness_mask14[euclidean_dist14 > FAR_THRESHOLD] = 1
        initial_mask = torch.ones((1, K)).cuda()

        aggregated_mask = (
            objectness_mask13 + objectness_mask12 + objectness_mask14 + initial_mask
        ) >= ACCEPTANCE_THRESHOLD
        _, aggregated_mask = np.nonzero(aggregated_mask.cpu().numpy())
        filtered_preds = np.array(proposals[0][b], dtype=np.object)[aggregated_mask]

        final_proposals.append(filtered_preds)

    return final_proposals


def find_matching_boxes_iou(proposals):
    """
    Args
    boxes1,boxes2: list of tuples -> (class,box corners, objectness_score)
    TODO:extend to batch
    """
    B = len(proposals[0])
    FAR_THRESHOLD = 0.6
    NEAR_THRESHOLD = 0.3
    ACCEPTANCE_THRESHOLD = 4
    IOU_THRESH = 0.25
    print("ACCEPTANCE THRESHOLD ", ACCEPTANCE_THRESHOLD)
    final_proposals = []
    proposals_inv = []
    for b in range(B):
        boxes1 = np.array([bx[1] for bx in proposals[0][b]])
        boxes2 = np.array([bx[1] for bx in proposals[1][b]])
        boxes3 = np.array([bx[1] for bx in proposals[2][b]])
        boxes4 = np.array([bx[1] for bx in proposals[3][b]])

        # K = boxes1.shape[1]
        K = boxes1.shape[0]
        iou_scores12 = nn_distance_iou(boxes1, boxes2)  # dist1: BxK, dist2: BxK2
        iou_scores13 = nn_distance_iou(boxes1, boxes3)  # dist1: BxK, dist2: BxK2
        iou_scores14 = nn_distance_iou(boxes1, boxes4)  # dist1: BxK, dist2: BxK2
        iou_mask12 = np.array(iou_scores12 >= IOU_THRESH, dtype=np.int)
        iou_mask13 = np.array(iou_scores13 >= IOU_THRESH, dtype=np.int)
        iou_mask14 = np.array(iou_scores14 >= IOU_THRESH, dtype=np.int)

        initial_mask = np.ones((1, K))

        bool_aggregated_mask = (
            iou_mask12 + iou_mask13 + iou_mask14 + initial_mask
        ) >= ACCEPTANCE_THRESHOLD

        _, aggregated_mask = np.nonzero(bool_aggregated_mask)
        _, aggregated_mask_inv = np.nonzero(np.logical_not(bool_aggregated_mask))
        # take ones from original
        filtered_preds = np.array(proposals[0][b], dtype=np.object)[aggregated_mask]
        eliminated_preds = np.array(proposals[0][b], dtype=np.object)[
            aggregated_mask_inv
        ]
        final_proposals.append(filtered_preds)
        proposals_inv.append(eliminated_preds)

    return final_proposals, proposals_inv


def dump_boxes(boxes, prefix=None):
    batch_size = len(boxes)

    for b in range(batch_size):
        boxes_to_be_dumped = []
        num_boxes = len(boxes[b])
        for j in range(num_boxes):
            boxes_to_be_dumped.append(
                pc_util.point_cloud_to_bbox(flip_axis_to_depth(boxes[b][j][1]))
            )

        boxes_to_be_dumped = np.array(boxes_to_be_dumped)
        if prefix is None:
            name = "scene"

        else:
            name = prefix

        dump = np.zeros((len(boxes_to_be_dumped), 7))
        dump[:, :6] = boxes_to_be_dumped
        pc_util.write_oriented_bbox(dump, name + str(b) + ".ply")


def aggregate_predictions(batch_pred_map_cls_multiple, rotations, dump_dir=None):
    batch_size = len(batch_pred_map_cls_multiple[0])
    # batch_size = 1
    angles = [0, 90, 180, 270]

    for b in range(batch_size):
        all_boxes = []
        for idx, rot in enumerate(rotations):
            all_boxes.append([])
            pred_boxes_in_scene = [
                box_tuple[1] for box_tuple in batch_pred_map_cls_multiple[idx][b]
            ]
            stacked_boxes = flip_axis_to_depth(np.vstack(pred_boxes_in_scene))
            # stacked_boxes = np.vstack(pred_boxes_in_scene)
            rotated_boxes, _ = pc_util.rotate_point_cloud(
                stacked_boxes, rotation_matrix=rot.T
            )  # unrotate stacked_boxes
            rotated_boxes = rotated_boxes.reshape((len(pred_boxes_in_scene), 8, 3))
            boxes_to_be_dumped = []

            for i, bx in enumerate(rotated_boxes):
                elem = batch_pred_map_cls_multiple[idx][b][i]
                bx = flip_axis_to_camera(bx)
                batch_pred_map_cls_multiple[idx][b][i] = (elem[0], bx, elem[2])

    # final_proposals = find_matching_boxes_center(batch_pred_map_cls_multiple[:])
    final_proposals, proposals_inv = find_matching_boxes_iou(
        batch_pred_map_cls_multiple[:]
    )
    if dump_dir is not None:
        dump_boxes(final_proposals, dump_dir + "/Selected")
        dump_boxes(proposals_inv, dump_dir + "/Discarded")
        dump_boxes(batch_pred_map_cls_multiple[0], dump_dir + "/0_")
        dump_boxes(batch_pred_map_cls_multiple[1], dump_dir + "/90_")
        dump_boxes(batch_pred_map_cls_multiple[2], dump_dir + "/180_")
        dump_boxes(batch_pred_map_cls_multiple[3], dump_dir + "/270_")
    # final_proposals = batch_pred_map_cls_multiple[0]
    # boxes_to_be_dumped = np.array(all_boxes[idx])
    # pc_util.write_bbox(boxes_to_be_dumped,str(idx) + ".ply")
    return final_proposals


class APCalculator(object):
    """Calculating Average Precision"""

    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()

    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert bsize == len(batch_gt_map_cls)
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self):
        """Use accumulated predictions and groundtruths to compute Average Precision."""
        rec, prec, ap = eval_det_multiprocessing(
            self.pred_map_cls,
            self.gt_map_cls,
            ovthresh=self.ap_iou_thresh,
            get_iou_func=get_iou_obb,
        )
        # rec, prec, ap,ious = eval_det_iou(self.pred_map_cls,
        #  self.gt_map_cls,
        #  ovthresh=self.ap_iou_thresh,
        #  get_iou_func=get_iou_obb)

        ret_dict = {}
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict["%s Average Precision" % (clsname)] = ap[key]
        ret_dict["mAP"] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict["%s Recall" % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict["%s Recall" % (clsname)] = 0
                rec_list.append(0)

        # for key in sorted(ious.keys()):
        #     clsname = self.class2type_map[key] if self.class2type_map else str(
        #         key)
        #     ret_dict['%s Average IOU' % (clsname)] = np.array(ious[key]).mean()
        ret_dict["AR"] = np.mean(rec_list)
        return ret_dict

        # In [1]

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0

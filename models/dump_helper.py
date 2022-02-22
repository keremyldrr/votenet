# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from numpy.lib.function_base import angle
import torch
import os
import sys
from varname import nameof
import trimesh
from models import ap_helper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
# from backbone_module import Pointet2Backbone

import pc_util
from matplotlib import cm

viridis = cm.get_cmap("jet", 64)
DUMP_CONF_THRESH = 0.5  # Dump boxes with obj prob larger than that.


def softmax(x):
    """Numpy function for softmax"""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def dump_only_boxes_gt(boxes, dump_dir):
    """
    pred_sem_cls[i, j].item(),
                          pred_corners_3d_upright_camera[i, j],
                          pred_variances[i, j],

    """
    for bidx, item in enumerate(boxes):
        for iidx, q in enumerate(item):
            cls, bbox = q
            # print("bidx", bidx)

            # print("cls", cls)
            # print("bbox", bbox)

            filename = "gt_{}_{}_{}.ply".format(bidx, iidx, cls)

            path = os.path.join(dump_dir, filename)
            ss = ap_helper.flip_axis_to_depth(bbox)
            trimesh.points.PointCloud(ss).convex_hull.export(path)


def dump_only_boxes(boxes, dump_dir):
    """
    pred_sem_cls[i, j].item(),
                          pred_corners_3d_upright_camera[i, j],
                          pred_variances[i, j],

    """
    N = 10  # num samples
    for bidx, item in enumerate(boxes):
        print(item[0])
        for iidx, q in enumerate(item):
            cls, var, bbox = q
            # print("bidx", bidx)

            # print("cls", cls)
            # print("var", var)
            # print("bbox", bbox)
            sampled_boxes = [bbox + np.random.randn(3) * var.item() for i in range(N)]

            for idx, ss in enumerate(sampled_boxes):
                filename = "{}_{}_{}_{}.ply".format(bidx, iidx, cls, idx)

                path = os.path.join(dump_dir, filename)
                ss = ap_helper.flip_axis_to_depth(ss)
                trimesh.points.PointCloud(ss).convex_hull.export(path)


def dump_results(end_points, dump_dir, config, inference_switch=False):
    """Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    """
    if not os.path.exists(dump_dir):
        os.system("mkdir %s" % (dump_dir))

    # INPUT
    point_clouds = end_points["point_clouds"].cpu().numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = end_points["seed_xyz"].detach().cpu().numpy()  # (B,num_seed,3)
    if "vote_xyz" in end_points:
        aggregated_vote_xyz = end_points["aggregated_vote_xyz"].detach().cpu().numpy()
        vote_xyz = end_points["vote_xyz"].detach().cpu().numpy()  # (B,num_seed,3)
        aggregated_vote_xyz = end_points["aggregated_vote_xyz"].detach().cpu().numpy()
    objectness_scores = (
        end_points["objectness_scores"].detach().cpu().numpy()
    )  # (B,K,2)
    pred_center = end_points["center"].detach().cpu().numpy()  # (B,K,3)
    pred_heading_class = torch.argmax(
        end_points["heading_scores"], -1
    )  # B,num_proposal
    pred_heading_residual = torch.gather(
        end_points["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)
    )  # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    pred_heading_residual = (
        pred_heading_residual.squeeze(2).detach().cpu().numpy()
    )  # B,num_proposal
    pred_size_class = torch.argmax(end_points["size_scores"], -1)  # B,num_proposal
    pred_size_residual = torch.gather(
        end_points["size_residuals"],
        2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
    )  # B,num_proposal,1,3
    pred_size_residual = (
        pred_size_residual.squeeze(2).detach().cpu().numpy()
    )  # B,num_proposal,3

    # OTHERS
    pred_mask = end_points["pred_mask"]  # B,num_proposal
    idx_beg = 0

    for i in range(batch_size):
        pc = point_clouds[i, :, :]
        objectness_prob = softmax(objectness_scores[i, :, :])[:, 1]  # (K,)

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, "%06d_pc.ply" % (idx_beg + i)))
        pc_util.write_ply(
            seed_xyz[i, :, :],
            os.path.join(dump_dir, "%06d_seed_pc.ply" % (idx_beg + i)),
        )
        if "vote_xyz" in end_points:
            pc_util.write_ply(
                end_points["vote_xyz"][i, :, :],
                os.path.join(dump_dir, "%06d_vgen_pc.ply" % (idx_beg + i)),
            )
            pc_util.write_ply(
                aggregated_vote_xyz[i, :, :],
                os.path.join(dump_dir, "%06d_aggregated_vote_pc.ply" % (idx_beg + i)),
            )
            pc_util.write_ply(
                aggregated_vote_xyz[i, :, :],
                os.path.join(dump_dir, "%06d_aggregated_vote_pc.ply" % (idx_beg + i)),
            )
        pc_util.write_ply(
            pred_center[i, :, 0:3],
            os.path.join(dump_dir, "%06d_proposal_pc.ply" % (idx_beg + i)),
        )
        if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
            pc_util.write_ply(
                pred_center[i, objectness_prob > DUMP_CONF_THRESH, 0:3],
                os.path.join(
                    dump_dir, "%06d_confident_proposal_pc.ply" % (idx_beg + i)
                ),
            )

        # Dump predicted bounding boxes
        if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                obb = config.param2obb(
                    pred_center[i, j, 0:3],
                    pred_heading_class[i, j],
                    pred_heading_residual[i, j],
                    pred_size_class[i, j],
                    pred_size_residual[i, j],
                )
                obbs.append(obb)
            if len(obbs) > 0:
                obbs = np.vstack(tuple(obbs))  # (num_proposal, 7)
                pc_util.write_oriented_bbox(
                    obbs[objectness_prob > DUMP_CONF_THRESH, :],
                    os.path.join(
                        dump_dir, "%06d_pred_confident_bbox.ply" % (idx_beg + i)
                    ),
                )
                pc_util.write_oriented_bbox(
                    obbs[
                        np.logical_and(
                            objectness_prob > DUMP_CONF_THRESH, pred_mask[i, :] == 1
                        ),
                        :,
                    ],
                    os.path.join(
                        dump_dir, "%06d_pred_confident_nms_bbox.ply" % (idx_beg + i)
                    ),
                )
                pc_util.write_oriented_bbox(
                    obbs[pred_mask[i, :] == 1, :],
                    os.path.join(dump_dir, "%06d_pred_nms_bbox.ply" % (idx_beg + i)),
                )
                pc_util.write_oriented_bbox(
                    obbs, os.path.join(dump_dir, "%06d_pred_bbox.ply" % (idx_beg + i))
                )

    # Return if it is at inference time. No dumping of groundtruths
    if inference_switch:
        return

    # LABELS
    gt_center = end_points["center_label"].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points["box_label_mask"].cpu().numpy()  # B,K2
    gt_heading_class = end_points["heading_class_label"].cpu().numpy()  # B,K2
    gt_heading_residual = end_points["heading_residual_label"].cpu().numpy()  # B,K2
    gt_size_class = end_points["size_class_label"].cpu().numpy()  # B,K2
    gt_size_residual = end_points["size_residual_label"].cpu().numpy()  # B,K2,3
    objectness_label = end_points["objectness_label"].detach().cpu().numpy()  # (B,K,)
    objectness_mask = end_points["objectness_mask"].detach().cpu().numpy()  # (B,K,)

    for i in range(batch_size):
        if np.sum(objectness_label[i, :]) > 0:
            pc_util.write_ply(
                pred_center[i, objectness_label[i, :] > 0, 0:3],
                os.path.join(
                    dump_dir, "%06d_gt_positive_proposal_pc.ply" % (idx_beg + i)
                ),
            )
        if np.sum(objectness_mask[i, :]) > 0:
            pc_util.write_ply(
                pred_center[i, objectness_mask[i, :] > 0, 0:3],
                os.path.join(dump_dir, "%06d_gt_mask_proposal_pc.ply" % (idx_beg + i)),
            )
        pc_util.write_ply(
            gt_center[i, :, 0:3],
            os.path.join(dump_dir, "%06d_gt_centroid_pc.ply" % (idx_beg + i)),
        )
        pc_util.write_ply_color(
            pred_center[i, :, 0:3],
            objectness_label[i, :],
            os.path.join(
                dump_dir, "%06d_proposal_pc_objectness_label.obj" % (idx_beg + i)
            ),
        )

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i, j] == 0:
                continue
            obb = config.param2obb(
                gt_center[i, j, 0:3],
                gt_heading_class[i, j],
                gt_heading_residual[i, j],
                gt_size_class[i, j],
                gt_size_residual[i, j],
            )
            obbs.append(obb)
        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(
                obbs, os.path.join(dump_dir, "%06d_gt_bbox.ply" % (idx_beg + i))
            )

    # OPTIONALL, also dump prediction and gt details
    if "batch_pred_map_cls" in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, "%06d_pred_map_cls.txt" % (ii)), "w")
            for t in end_points["batch_pred_map_cls"][ii]:
                fout.write(str(t[0]) + " ")
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(" " + str(t[2]))
                fout.write("\n")
            fout.close()
    if "batch_gt_map_cls" in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, "%06d_gt_map_cls.txt" % (ii)), "w")
            for t in end_points["batch_gt_map_cls"][ii]:
                fout.write(str(t[0]) + " ")
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write("\n")
            fout.close()


def dump_results_mini(end_points, dump_dir, config, inference_switch=False):
    """Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    """
    if not os.path.exists(dump_dir):
        os.system("mkdir %s" % (dump_dir))

    # INPUT
    point_clouds = end_points["point_clouds"].cpu().numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = end_points["seed_xyz"].detach().cpu().numpy()  # (B,num_seed,3)
    if "vote_xyz" in end_points:
        aggregated_vote_xyz = end_points["aggregated_vote_xyz"].detach().cpu().numpy()
        vote_xyz = end_points["vote_xyz"].detach().cpu().numpy()  # (B,num_seed,3)
        aggregated_vote_xyz = end_points["aggregated_vote_xyz"].detach().cpu().numpy()
    objectness_scores = (
        end_points["objectness_scores"].detach().cpu().numpy()
    )  # (B,K,2)
    pred_center = end_points["center"].detach().cpu().numpy()  # (B,K,3)
    pred_heading_class = torch.argmax(
        end_points["heading_scores"], -1
    )  # B,num_proposal
    pred_heading_residual = torch.gather(
        end_points["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)
    )  # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    pred_heading_residual = (
        pred_heading_residual.squeeze(2).detach().cpu().numpy()
    )  # B,num_proposal
    pred_size_class = torch.argmax(end_points["size_scores"], -1)  # B,num_proposal
    pred_size_residual = torch.gather(
        end_points["size_residuals"],
        2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
    )  # B,num_proposal,1,3
    pred_size_residual = (
        pred_size_residual.squeeze(2).detach().cpu().numpy()
    )  # B,num_proposal,3

    # OTHERS
    pred_mask = end_points["pred_mask"]  # B,num_proposal
    idx_beg = 0

    for i in range(batch_size):
        pc = point_clouds[i, :, :]
        objectness_prob = softmax(objectness_scores[i, :, :])[:, 1]  # (K,)

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, "%06d_pc.ply" % (idx_beg + i)))

        pc_util.write_ply(
            pred_center[i, :, 0:3],
            os.path.join(dump_dir, "%06d_proposal_pc.ply" % (idx_beg + i)),
        )
        if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
            pc_util.write_ply(
                pred_center[i, objectness_prob > DUMP_CONF_THRESH, 0:3],
                os.path.join(
                    dump_dir, "%06d_confident_proposal_pc.ply" % (idx_beg + i)
                ),
            )

        # Dump predicted bounding boxes
        if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                obb = config.param2obb(
                    pred_center[i, j, 0:3],
                    pred_heading_class[i, j],
                    pred_heading_residual[i, j],
                    pred_size_class[i, j],
                    pred_size_residual[i, j],
                )
                obbs.append(obb)
            if len(obbs) > 0:
                obbs = np.vstack(tuple(obbs))  # (num_proposal, 7)

                pc_util.write_oriented_bbox(
                    obbs[
                        np.logical_and(
                            objectness_prob > DUMP_CONF_THRESH, pred_mask[i, :] == 1
                        ),
                        :,
                    ],
                    os.path.join(
                        dump_dir, "%06d_pred_confident_nms_bbox.ply" % (idx_beg + i)
                    ),
                )

    # Return if it is at inference time. No dumping of groundtruths
    if inference_switch:
        return

    # LABELS
    gt_center = end_points["center_label"].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points["box_label_mask"].cpu().numpy()  # B,K2
    gt_heading_class = end_points["heading_class_label"].cpu().numpy()  # B,K2
    gt_heading_residual = end_points["heading_residual_label"].cpu().numpy()  # B,K2
    gt_size_class = end_points["size_class_label"].cpu().numpy()  # B,K2
    gt_size_residual = end_points["size_residual_label"].cpu().numpy()  # B,K2,3
    objectness_label = end_points["objectness_label"].detach().cpu().numpy()  # (B,K,)
    objectness_mask = end_points["objectness_mask"].detach().cpu().numpy()  # (B,K,)

    for i in range(batch_size):
        if np.sum(objectness_label[i, :]) > 0:
            pc_util.write_ply(
                pred_center[i, objectness_label[i, :] > 0, 0:3],
                os.path.join(
                    dump_dir, "%06d_gt_positive_proposal_pc.ply" % (idx_beg + i)
                ),
            )
        if np.sum(objectness_mask[i, :]) > 0:
            pc_util.write_ply(
                pred_center[i, objectness_mask[i, :] > 0, 0:3],
                os.path.join(dump_dir, "%06d_gt_mask_proposal_pc.ply" % (idx_beg + i)),
            )
        pc_util.write_ply(
            gt_center[i, :, 0:3],
            os.path.join(dump_dir, "%06d_gt_centroid_pc.ply" % (idx_beg + i)),
        )
        pc_util.write_ply_color(
            pred_center[i, :, 0:3],
            objectness_label[i, :],
            os.path.join(
                dump_dir, "%06d_proposal_pc_objectness_label.obj" % (idx_beg + i)
            ),
        )

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i, j] == 0:
                continue
            obb = config.param2obb(
                gt_center[i, j, 0:3],
                gt_heading_class[i, j],
                gt_heading_residual[i, j],
                gt_size_class[i, j],
                gt_size_residual[i, j],
            )
            obbs.append(obb)
        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(
                obbs, os.path.join(dump_dir, "%06d_gt_bbox.ply" % (idx_beg + i))
            )

    # OPTIONALL, also dump prediction and gt details
    if "batch_pred_map_cls" in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, "%06d_pred_map_cls.txt" % (ii)), "w")
            for t in end_points["batch_pred_map_cls"][ii]:
                fout.write(str(t[0]) + " ")
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(" " + str(t[2]))
                fout.write("\n")
            fout.close()
    if "batch_gt_map_cls" in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, "%06d_gt_map_cls.txt" % (ii)), "w")
            for t in end_points["batch_gt_map_cls"][ii]:
                fout.write(str(t[0]) + " ")
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write("\n")
            fout.close()


def dump_results_for_sanity_check(
    end_points, dump_dir, config, inference_switch=False, prefix=""
):
    """Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    """
    if not os.path.exists(dump_dir):
        os.system("mkdir %s" % (dump_dir))

    # INPUT
    pcds = end_points["point_clouds"].cpu()  # .numpy()
    batch_size = pcds.shape[0]
    # NETWORK OUTPUTS

    # OTHERS
    idx_beg = 0
    point_clouds = []
    for idx, p in enumerate(pcds):
        point_clouds.append(p[:, 0:3].contiguous().numpy())

    if inference_switch:
        return

    # LABELS
    gt_center = end_points["center_label"].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points["box_label_mask"].cpu().numpy()  # B,K2
    gt_heading_class = end_points["heading_class_label"].cpu().numpy()  # B,K2
    gt_heading_residual = end_points["heading_residual_label"].cpu().numpy()  # B,K2
    gt_size_class = end_points["size_class_label"].cpu().numpy()  # B,K2
    gt_size_residual = end_points["size_residual_label"].cpu().numpy()  # B,K2,3

    # angles = [np.pi/2,np.pi]
    # rotations = [np.eye(3)] + [pc_util.rotx(a) for a in angles] + [pc_util.roty(a) for a in angles] + [pc_util.rotz(a) for a in angles]

    for i in range(batch_size):

        # Dump GT bounding boxes
        pc_util.write_ply(point_clouds[i], os.path.join(dump_dir, "%06d_pc.ply" % (i)))

        # pc_util.write_ply(rotated, os.path.join(dump_dir, '%06d_pc_rotated_%d.ply'%(idx_beg + i,idx)))
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i, j] == 0:
                continue
            # center is rotated here if needed
            # print("HC HR SC SR",gt_heading_class[i,j], gt_heading_residual[i,j],gt_size_class[i,j],gt_size_residual[i,j])
            obb = config.param2obb(
                gt_center[i, j, 0:3],
                gt_heading_class[i, j],
                gt_heading_residual[i, j],
                gt_size_class[i, j],
                gt_size_residual[i, j],
            )
            obbs.append(obb)
        if len(obbs) > 0:
            p_obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            # pass rotmat here for rotation
            pc_util.write_oriented_bbox(
                p_obbs,
                os.path.join(
                    dump_dir, "%s_%06d_gt_bbox_%d.ply" % (prefix, idx_beg + i, idx)
                ),
            )
            # pc_util.scene_snapshot(point_clouds[i],p_obbs,os.path.join(dump_dir, '%s_%06d_gt_bbox_%d.png'%(prefix,idx_beg+i,idx)))


def dump_results_with_color(end_points, dump_dir, config, inference_switch=False):
    """Dump results.

    Args:
        samples: list of dicts
            {...,cls_entropy,objetness_entropy,box_size_variance,iou_mask, pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    """
    if not os.path.exists(dump_dir):
        os.system("mkdir %s" % (dump_dir))

    # INPUT
    point_clouds = end_points["point_clouds"].cpu().numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = end_points["seed_xyz"].detach().cpu().numpy()  # (B,num_seed,3)
    if "vote_xyz" in end_points:
        aggregated_vote_xyz = end_points["aggregated_vote_xyz"].detach().cpu().numpy()
        vote_xyz = end_points["vote_xyz"].detach().cpu().numpy()  # (B,num_seed,3)
        aggregated_vote_xyz = end_points["aggregated_vote_xyz"].detach().cpu().numpy()
    objectness_scores = (
        end_points["objectness_scores"].detach().cpu().numpy()
    )  # (B,K,2)
    pred_center = end_points["center"].detach().cpu().numpy()  # (B,K,3)
    pred_heading_class = torch.argmax(
        end_points["heading_scores"], -1
    )  # B,num_proposal
    pred_heading_residual = torch.gather(
        end_points["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)
    )  # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    pred_heading_residual = (
        pred_heading_residual.squeeze(2).detach().cpu().numpy()
    )  # B,num_proposal
    pred_size_class = torch.argmax(end_points["size_scores"], -1)  # B,num_proposal
    pred_size_residual = torch.gather(
        end_points["size_residuals"],
        2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3),
    )  # B,num_proposal,1,3
    pred_size_residual = (
        pred_size_residual.squeeze(2).detach().cpu().numpy()
    )  # B,num_proposal,3

    # OTHERS
    # TODO: For now, only single scene in a batch. Code is kind of mixed
    pred_mask = end_points["pred_mask"]  # B,num_proposal
    idx_beg = 0
    cls_entropy = end_points["cls_entropy"]
    obj_entropy = end_points["obj_entropy"]
    box_size_var = end_points["box_size_var"]
    iou_mask = end_points["iou_mask"]
    cls_entropy_color_map = viridis(
        (cls_entropy - cls_entropy.min()) / (cls_entropy.max() - cls_entropy.min())
    )
    obj_entropy_color_map = viridis(
        (obj_entropy - obj_entropy.min()) / (obj_entropy.max() - obj_entropy.min())
    )
    boxsize_entropy_color_map = viridis(
        (box_size_var - box_size_var.min()) / (box_size_var.max() - box_size_var.min())
    )
    for i in range(batch_size):
        pc = point_clouds[i, :, :]
        objectness_prob = softmax(objectness_scores[i, :, :])[:, 1]  # (K,)
        obj_prob_map = viridis(objectness_prob)
        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, "%06d_pc.ply" % (idx_beg + i)))

        # Dump predicted bounding boxes
        # if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
        num_proposal = pred_center.shape[1]
        names = ["cls_entropy", "obj_entropy", "box_size_var", "objectness_probability"]
        for idx, colormap in enumerate(
            [
                cls_entropy_color_map,
                obj_entropy_color_map,
                boxsize_entropy_color_map,
                obj_prob_map,
            ]
        ):
            obbs = []
            print(names[idx])
            for j in range(num_proposal):
                obb = config.param2colorobb(
                    pred_center[i, j, 0:3],
                    pred_heading_class[i, j],
                    pred_heading_residual[i, j],
                    pred_size_class[i, j],
                    pred_size_residual[i, j],
                    colormap[j],
                )
                obbs.append(obb)
            if len(obbs) > 0:
                obbs = np.vstack(tuple(obbs))  # (num_proposal, 7)
                pc_util.write_oriented_bbox_with_color(
                    obbs[objectness_prob > DUMP_CONF_THRESH, :],
                    os.path.join(
                        dump_dir,
                        names[idx] + "%06d_pred_confident_bbox.ply" % (idx_beg + i),
                    ),
                )
                pc_util.write_oriented_bbox_with_color(
                    obbs[
                        np.logical_and(
                            objectness_prob > DUMP_CONF_THRESH, pred_mask[i, :] == 1
                        ),
                        :,
                    ],
                    os.path.join(
                        dump_dir,
                        names[idx] + "%06d_pred_confident_nms_bbox.ply" % (idx_beg + i),
                    ),
                )
                pc_util.write_oriented_bbox_with_color(
                    obbs[pred_mask[i, :] == 1, :],
                    os.path.join(
                        dump_dir, names[idx] + "%06d_pred_nms_bbox.ply" % (idx_beg + i)
                    ),
                )
                pc_util.write_oriented_bbox_with_color(
                    obbs,
                    os.path.join(
                        dump_dir, names[idx] + "%06d_pred_bbox.ply" % (idx_beg + i)
                    ),
                )
                pc_util.write_oriented_bbox_with_color(
                    obbs[iou_mask == 1, :],
                    os.path.join(
                        dump_dir,
                        names[idx] + "%06d_iou_masked_bbox.ply" % (idx_beg + i),
                    ),
                )

    # Return if it is at inference time. No dumping of groundtruths
    if inference_switch:
        return

    # LABELS
    gt_center = end_points["center_label"].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points["box_label_mask"].cpu().numpy()  # B,K2
    gt_heading_class = end_points["heading_class_label"].cpu().numpy()  # B,K2
    gt_heading_residual = end_points["heading_residual_label"].cpu().numpy()  # B,K2
    gt_size_class = end_points["size_class_label"].cpu().numpy()  # B,K2
    gt_size_residual = end_points["size_residual_label"].cpu().numpy()  # B,K2,3
    objectness_label = end_points["objectness_label"].detach().cpu().numpy()  # (B,K,)
    objectness_mask = end_points["objectness_mask"].detach().cpu().numpy()  # (B,K,)

    for i in range(batch_size):
        if np.sum(objectness_label[i, :]) > 0:
            pc_util.write_ply(
                pred_center[i, objectness_label[i, :] > 0, 0:3],
                os.path.join(
                    dump_dir, "%06d_gt_positive_proposal_pc.ply" % (idx_beg + i)
                ),
            )
        if np.sum(objectness_mask[i, :]) > 0:
            pc_util.write_ply(
                pred_center[i, objectness_mask[i, :] > 0, 0:3],
                os.path.join(dump_dir, "%06d_gt_mask_proposal_pc.ply" % (idx_beg + i)),
            )
        pc_util.write_ply(
            gt_center[i, :, 0:3],
            os.path.join(dump_dir, "%06d_gt_centroid_pc.ply" % (idx_beg + i)),
        )
        pc_util.write_ply_color(
            pred_center[i, :, 0:3],
            objectness_label[i, :],
            os.path.join(
                dump_dir, "%06d_proposal_pc_objectness_label.obj" % (idx_beg + i)
            ),
        )

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i, j] == 0:
                continue
            obb = config.param2obb(
                gt_center[i, j, 0:3],
                gt_heading_class[i, j],
                gt_heading_residual[i, j],
                gt_size_class[i, j],
                gt_size_residual[i, j],
            )
            obbs.append(obb)
        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(
                obbs, os.path.join(dump_dir, "%06d_gt_bbox.ply" % (idx_beg + i))
            )

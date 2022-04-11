# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection with SUN RGB-D or ScanNet.

Sample usage:
python train.py --dataset sunrgbd --log_dir log_sunrgbd

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""
# Python stuff
import os
import sys
import time
import numpy as np
from datetime import datetime
import importlib.util
import argparse
import matplotlib.pyplot as plt
from numpy.random.mtrand import set_state

# pytorch stuff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch_lr_finder import LRFinder

BASE_DIR = os.path.dirname(
    os.path.abspath("/home/yildirir/workspace/votenet/README.md")
)
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))
sys.path.append(os.path.join(ROOT_DIR, "models"))

# project stuff

from ap_helper import APCalculator, parse_predictions, parse_groundtruths
from initialization_utils import initialize_dataloader, initialize_model, log_string


def get_current_lr(epoch, FLAGS):

    lr = FLAGS.LEARNING_RATE
    for i, lr_decay_epoch in enumerate(FLAGS.LR_DECAY_STEPS.split(",")):
        if epoch >= int(lr_decay_epoch):
            lr *= float(FLAGS.LR_DECAY_RATES.split(",")[i])
    return lr


def get_current_lr_by_iters(epoch, FLAGS):

    lr = FLAGS.LEARNING_RATE
    if FLAGS.FIXED_LR == 0:
        START_ITER = FLAGS.START_ITER
        ITER_DECAY_STEPS = [12000, 18000, 24000, 30000, 36000, 42000]
        DATA_SIZE = len(FLAGS.TRAIN_DATALOADER)
        ITERS_PER_EPOCH = DATA_SIZE
        NUM_ITERS_TOTAL = START_ITER + ITERS_PER_EPOCH * epoch

        # print(NUM_ITERS_TOTAL, ITER_DECAY_STEPS, ITERS_PER_EPOCH, DATA_SIZE)
        for i, lr_decay_epoch in enumerate(ITER_DECAY_STEPS):
            # print(NUM_ITERS_TOTAL, lr_decay_epoch)
            if NUM_ITERS_TOTAL >= lr_decay_epoch:
                lr *= 0.1
                # print("Shrank")

    return lr


def adjust_learning_rate(optimizer, epoch, FLAGS):

    # lr = get_current_lr(epoch)
    if FLAGS.FIXED_LR == 0:
        if FLAGS.START_ITER != 0:
            lr = get_current_lr_by_iters(epoch, FLAGS)
            # print("ITERS")
        else:
            # lr = get_current_lr(epoch, FLAGS)
            lr = get_current_lr_by_iters(epoch, FLAGS)
    else:
        lr = FLAGS.FIXED_LR
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ------------------------------------------------------------------------- GLOBAL CONFIG END*


def train_one_epoch(net, optimizer, criterion, bnm_scheduler, FLAGS):
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT, FLAGS)
    bnm_scheduler.step()  # decay BN momentum
    net.train()  # set model to training mode
    #
    # lr_finder = CustomLRFinder(net, optimizer, criterion, FLAGS.DEVICE)
    # lr_finder.range_test(FLAGS.TRAIN_DATALOADER, end_lr=10, num_iter=1000)
    # lr_finder.plot()
    # plt.savefig("LRvsLoss.png")
    # plt.close()
    for batch_idx, batch_data_label in enumerate(FLAGS.TRAIN_DATALOADER):
        # for key in batch_data_label:
        #     batch_data_label[key] = batch_data_label[key].to(FLAGS.DEVICE)
        #     i

        adjust_learning_rate(optimizer, EPOCH_CNT, FLAGS)
        # Forward pass
        for key in batch_data_label:
            if key != "name":
                batch_data_label[key] = batch_data_label[key].to(FLAGS.DEVICE)

            # Forward pass

        curr_lr = optimizer.param_groups[0]["lr"]

        optimizer.zero_grad()
        if "name" not in batch_data_label.keys():
            inputs = {"point_clouds": batch_data_label["point_clouds"]}
        else:

            inputs = {
                "point_clouds": batch_data_label["point_clouds"],
                "center_label": batch_data_label["center_label"],
                "name": batch_data_label["name"],
            }

        st = time.time()
        end_points = net(inputs)

        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert key not in end_points
            end_points[key] = batch_data_label[key]

        L1_reg = torch.tensor(0.0, requires_grad=True).cuda()
        for name, param in net.named_parameters():
            if "weight" in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        loss, end_points = criterion(end_points, FLAGS.DATASET_CONFIG)

        # end_points["loss"] += L1_reg * 1e-4
        # loss += L1_reg * 1e-4
        loss.backward()
        optimizer.step()
        # # pdb.set_trace()
        # if FLAGS.OVERFIT:
        #     print("ADDED SIGMA ", FLAGS.SIGMA)
        #     for bdx, log_vars_frame in enumerate(end_points["log_var_per_gt"]):
        #         for cdx, (box, sc) in enumerate(log_vars_frame):
        #             sigma = torch.exp(box.mean()) ** 0.5
        #             print(
        #                 bdx,
        #                 cdx,
        #                 sigma.item(),
        #                 np.unique(sc.detach().cpu(), return_counts=True),
        #             )
        # end_points["sigma"] = torch.exp(end_points["log_var"].mean())**0.5
        # Accumulate statistics and print out
        for key in end_points:
            if (
                "nll" in key
                or "sigma" in key
                or "loss" in key
                or "acc" in key
                or "ratio" in key
            ):

                # print(key, end_points[key].shape)
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 1 if FLAGS.OVERFIT == True else 10
        if (batch_idx + 1) % batch_interval == 0:
            log_string(FLAGS.LOGGER, " ---- batch: %03d ----" % (batch_idx + 1))
            FLAGS.TRAIN_VISUALIZER.log_scalars(
                {key: stat_dict[key] / batch_interval for key in stat_dict},
                step=(EPOCH_CNT * len(FLAGS.TRAIN_DATALOADER) + batch_idx)
                * FLAGS.BATCH_SIZE,
            )
            FLAGS.TRAIN_VISUALIZER.log_scalars(
                {"lr": curr_lr},
                step=(EPOCH_CNT * len(FLAGS.TRAIN_DATALOADER) + batch_idx)
                * FLAGS.BATCH_SIZE,
            )
            for key in sorted(stat_dict.keys()):
                log_string(
                    FLAGS.LOGGER, "mean %s: %f" % (key, stat_dict[key] / batch_interval)
                )
                stat_dict[key] = 0


def evaluate_one_epoch(net, criterion, FLAGS):
    global best_map

    # Used for AP calculation during evaluation
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
    for batch_idx, batch_data_label in enumerate(FLAGS.TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print("Eval batch: %d" % (batch_idx))
        for key in batch_data_label:
            if key != "name":
                batch_data_label[key] = batch_data_label[key].to(FLAGS.DEVICE)

        # Forward pass
        if "name" not in batch_data_label.keys():
            inputs = {"point_clouds": batch_data_label["point_clouds"]}
        else:

            inputs = {
                "point_clouds": batch_data_label["point_clouds"],
                "center_label": batch_data_label["center_label"],
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

        L1_reg = torch.tensor(0.0, requires_grad=False).cuda()
        for name, param in net.named_parameters():
            if "weight" in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        # Accumulate statistics and prin t out
        # t ot
        # end_points["loss"] += L1_reg * 1e-4
        loss = end_points["loss"]
        for key in end_points:
            if "loss" in key or "acc" in key or "ratio" in key or "nll" in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        # batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        # batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        # ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        if FLAGS.DUMP_RESULTS and batch_idx == 0 and EPOCH_CNT % 10 == 0:
            FLAGS.MODEL.DUMP_RESULTS(end_points, FLAGS.DUMP_DIR, FLAGS.DATASET_CONFIG)
        if FLAGS.NUM_VAL_BATCHES != -1:
            if batch_idx > FLAGS.NUM_VAL_BATCHES:
                best_map = -1
                break
    # Log statistics

    FLAGS.TEST_VISUALIZER.log_scalars(
        {key: stat_dict[key] / float(batch_idx + 1) for key in stat_dict},
        (EPOCH_CNT + 1) * len(FLAGS.TRAIN_DATALOADER) * FLAGS.BATCH_SIZE,
    )

    for key in sorted(stat_dict.keys()):
        log_string(
            FLAGS.LOGGER,
            "eval mean %s: %f" % (key, stat_dict[key] / (float(batch_idx + 1))),
        )

    # Evaluate average precision
    # metrics_dict = ap_calculator.compute_metrics()
    # for key in metrics_dict:
    #     log_string(FLAGS.LOGGER, "eval %s: %f" % (key, metrics_dict[key]))

    mean_loss = stat_dict["loss"] / float(batch_idx + 1)
    return mean_loss, mean_loss  # metrics_dict["mAP"]


def train(FLAGS):
    global EPOCH_CNT
    min_loss = 1e10
    loss = 0
    best_map = 0
    curr_map = 0
    initialize_dataloader(FLAGS)

    net, criterion, optimizer, bnm_scheduler = initialize_model(FLAGS)
    # TODO: initialize everything here and create optimizer, net and criterion here, then pass it to the corresponding functions
    for epoch in range(FLAGS.START_EPOCH, FLAGS.MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string(FLAGS.LOGGER, "**** EPOCH %03d ****" % (epoch))
        if FLAGS.START_ITER != 0:
            log_string(
                FLAGS.LOGGER,
                "Current learning rate: %f" % (get_current_lr_by_iters(epoch, FLAGS)),
            )
        else:
            log_string(
                FLAGS.LOGGER,
                "Current learning rate: %f" % (get_current_lr(epoch, FLAGS)),
            )
        log_string(
            FLAGS.LOGGER,
            "Current BN decay momentum: %f"
            % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)),
        )
        log_string(FLAGS.LOGGER, str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorc /pytorch/issues/5059
        np.random.seed()
        train_one_epoch(net, optimizer, criterion, bnm_scheduler, FLAGS)
        save_interval = 3 if (FLAGS.OVERFIT == False) else 100
        if EPOCH_CNT % save_interval == 1:  # Eval every 2 epochs
            # TODO: make this dataset dependent, or iterations
            DATA_SIZE = len(FLAGS.TRAIN_DATALOADER)
            ITERS_PER_EPOCH = DATA_SIZE

            NUM_ITERS_TOTAL = FLAGS.START_ITER + ITERS_PER_EPOCH * epoch
            save_dict = {
                "epoch": epoch
                + 1,  # after training one epoch, the start_epoch should be epoch+1
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }
            try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict["model_state_dict"] = net.module.state_dict()
            except:
                save_dict["model_state_dict"] = net.state_dict()

            torch.save(
                save_dict,
                os.path.join(
                    FLAGS.LOG_DIR, "checkpoint_{}.tar".format(NUM_ITERS_TOTAL)
                ),
            )  #

            loss, curr_map = evaluate_one_epoch(net, criterion, FLAGS)
            FLAGS.TEST_VISUALIZER.log_scalars(
                {"mAP": curr_map},
                (EPOCH_CNT + 1) * len(FLAGS.TRAIN_DATALOADER) * FLAGS.BATCH_SIZE,
            )

            # Save checkpoint
            if curr_map < best_map:
                DATA_SIZE = len(FLAGS.TRAIN_DATALOADER)
                ITERS_PER_EPOCH = DATA_SIZE
                NUM_ITERS_TOTAL = FLAGS.START_ITER + ITERS_PER_EPOCH * epoch
                torch.save(
                    save_dict,
                    os.path.join(
                        FLAGS.LOG_DIR, "best_checkpoint_{}.tar".format(NUM_ITERS_TOTAL)
                    ),
                )  # FIXME re enable
                log_string(
                    FLAGS.LOGGER,
                    "Best changed from {} to {}".format(best_map, curr_map),
                )
                best_map = curr_map
            else:

                log_string(FLAGS.LOGGER, "No change, best is {}".format(best_map))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("C", args.config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    FLAGS = mod.C

    train(FLAGS)

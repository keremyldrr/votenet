# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" Evaluation routine for 3D object detection with SUN RGB-D and ScanNet.
"""
#!/usr/bin/env python

#%%
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
from utils.uncertainty_utils import box_size_uncertainty, semantic_cls_uncertainty , objectness_uncertainty,center_uncertainty , apply_softmax,compute_objectness_accuracy,compute_iou_masks
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from ap_helper import APCalculator, parse_predictions, parse_groundtruths
#%%


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

FLAGS = AttrDict()
FLAGS.ap_iou_thresholds='0.25,0.5'
FLAGS.batch_size=1
FLAGS.checkpoint_path='logs/log_scannet_dropout_0_1_200/checkpoint129.tar'
# FLAGS.checkpoint_path='logs/log_scannet120/checkpoint.tar'
FLAGS.cluster_sampling='seed_fps'
FLAGS.conf_thresh=0.5
FLAGS.dataset='scannet'
FLAGS.dump_dir='evals/test_colors'
FLAGS.faster_eval=False
FLAGS.model='votenet'
FLAGS.nms_iou=0.25
FLAGS.no_height=False
FLAGS.num_point=40000
FLAGS.num_target=256
FLAGS.per_class_proposal=False
FLAGS.shuffle_dataset=False
FLAGS.use_3d_nms=True
FLAGS.use_cls_nms=True
FLAGS.use_color=False
FLAGS.use_old_type_nms=False
FLAGS.use_sunrgbd_v2=False
FLAGS.vote_factor=1

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DUMP_DIR = FLAGS.dump_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path
assert (CHECKPOINT_PATH is not None)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]
NUM_SAMPLES = 5
NUM_SCENES = 1
# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    DUMP_FOUT.write(out_str + '\n')
    DUMP_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
    from model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    TEST_DATASET = SunrgbdDetectionVotesDataset(
        'val',
        num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color,
        use_height=(not FLAGS.no_height),
        use_v1=(not FLAGS.use_sunrgbd_v2))
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    TEST_DATASET = ScannetDetectionDataset('val',
                                           num_points=NUM_POINT,
                                           augment=False,
                                           use_color=FLAGS.use_color,
                                           use_height=(not FLAGS.no_height))
else:
    print('Unknown dataset %s. Exiting...' % (FLAGS.dataset))
    exit(-1)
print(len(TEST_DATASET))
TEST_DATALOADER = DataLoader(TEST_DATASET,
                             batch_size=BATCH_SIZE,
                             shuffle=FLAGS.shuffle_dataset,
                             num_workers=4,
                             worker_init_fn=my_worker_init_fn)

# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model)  # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1

if FLAGS.model == 'boxnet':
    Detector = MODEL.BoxNet
else:
    Detector = MODEL.VoteNet

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)
net.to(device)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load checkpoint if there is any
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    log_string("Loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, epoch))

# Used for AP calculation
CONFIG_DICT = {
    'remove_empty_box': (not FLAGS.faster_eval),
    'use_3d_nms': FLAGS.use_3d_nms,
    'nms_iou': FLAGS.nms_iou,
    'use_old_type_nms': FLAGS.use_old_type_nms,
    'cls_nms': FLAGS.use_cls_nms,
    'per_class_proposal': FLAGS.per_class_proposal,
    'conf_thresh': FLAGS.conf_thresh,
    'dataset_config': DATASET_CONFIG
}
# ------------------------------------------------------------------------- GLOBAL CONFIG END


def evaluate_one_epoch():
    stat_dict = {}
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
        for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d' % (batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        
        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        if batch_idx == 2:
            a = 1
            # MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG)
            # break
            

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f' % (key, stat_dict[key] /
                                         (float(batch_idx + 1))))

    # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-' * 10, 'iou_thresh: %f' % (AP_IOU_THRESHOLDS[i]), '-' * 10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f' % (key, metrics_dict[key]))

    mean_loss = stat_dict['loss'] / float(batch_idx + 1)
    return mean_loss

def evaluate_one_epoch_with_uncertainties():
    stat_dict = {}
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
        for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval()  # set model to eval mode (for bn and dp)
    net.pnet.drop1.train()
    net.pnet.drop2.train()
    net.pnet.drop3.train()    
    stats = []

    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d' % (batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        loss = np.zeros([NUM_SAMPLES])
        with torch.no_grad():
            mc_samples = [net(inputs) for i in range(NUM_SAMPLES)]

        # Compute loss
        obj_accs = []
        cls_accs = []
        size_accs = []
        obj_cls_accs = []
        obj_size_accs = []
        size_cls_accs = []
        obj_cls_size_accs = []
        for idx,end_points in enumerate(mc_samples):
            for key in batch_data_label:
                assert (key not in end_points)
                end_points[key] = batch_data_label[key]
            local_loss, end_points = criterion(end_points, DATASET_CONFIG)
            loss[idx] = local_loss.cpu().detach().numpy().item()
        #     print(loss)
            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        apply_softmax(mc_samples)
        print("Softmaxed")
        # mean_obj_acc = compute_objectness_accuracy(mc_samples)
        print("Obj accuracy computed")
        box_size_mask,box_size_var = box_size_uncertainty(mc_samples)
        print("Box size entropy computed")
        cls_entropy_mask,cls_entropy = semantic_cls_uncertainty(mc_samples)
        print("Semantic cls entropy computed")
        obj_entropy_mask,obj_entropy = objectness_uncertainty(mc_samples)
        print("Objectness entropy computed")
        center_unc_mask,center_unc = center_uncertainty(mc_samples)
        print("Center entropy computed")
        iou_masks = compute_iou_masks(mc_samples) #TODO: make this like a voting mask
        print("Iou masks computed")

        print("Obj entropy accuracy ",(256 -  np.sum(np.logical_xor(obj_entropy_mask,iou_masks[0])))/256)
        print("Cls entropy accuracy ", (256 - np.sum(np.logical_xor(cls_entropy_mask,iou_masks[0])))/256)
        #print("Center entropy accuracy ", np.sum(np.logical_and(center_unc_mask,iou_masks[0]))/256)
        print("Box size entropy accuracy ",(256 -  np.sum(np.logical_xor(box_size_mask,iou_masks[0])))/256)
        
        
        print("Obj cls entropy accuracy ",((256 -  np.sum(np.logical_xor(np.logical_and(obj_entropy_mask,cls_entropy_mask),iou_masks[0])))/256))
        print("Obj size entropy accuracy ", ((256 -  np.sum(np.logical_xor(np.logical_and(obj_entropy_mask,box_size_mask),iou_masks[0])))/256))
        #print("Center entropy accuracy ", np.sum(np.logical_and(center_unc_mask,iou_masks[0]))/256)
        print("Cls size entropy accuracy ",((256 -  np.sum(np.logical_xor(np.logical_and(box_size_mask,cls_entropy_mask),iou_masks[0])))/256))
        print("Obj cls Box size entropy accuracy ",((256 -  np.sum(np.logical_xor(np.logical_and(box_size_mask,np.logical_and(obj_entropy_mask,cls_entropy_mask)),iou_masks[0])))/256))

        obj_accs.append((256 -  np.sum(np.logical_xor(obj_entropy_mask,iou_masks[0])))/256)
        cls_accs.append((256 - np.sum(np.logical_xor(cls_entropy_mask,iou_masks[0])))/256)
        size_accs.append((256 -  np.sum(np.logical_xor(box_size_mask,iou_masks[0])))/256)

        obj_cls_accs.append((256 -  np.sum(np.logical_xor(np.logical_and(obj_entropy_mask,cls_entropy_mask),iou_masks[0])))/256)
        obj_size_accs.append((256 -  np.sum(np.logical_xor(np.logical_and(obj_entropy_mask,box_size_mask),iou_masks[0])))/256)
        size_cls_accs.append((256 -  np.sum(np.logical_xor(np.logical_and(box_size_mask,cls_entropy_mask),iou_masks[0])))/256)
        obj_cls_size_accs.append((256 -  np.sum(np.logical_xor(np.logical_and(box_size_mask,np.logical_and(obj_entropy_mask,cls_entropy_mask)),iou_masks[0])))/256)
        # for idx,e in enumerate(mc_samples):
        #     e["cls_entropy"] =  cls_entropy
        #     e["obj_entropy"] =  obj_entropy
        #     e["box_size_var"] =  box_size_var
        #     e["iou_mask"] = iou_masks[idx]
        #     MODEL.dump_results_with_color(e, DUMP_DIR + str(idx), DATASET_CONFIG)

                # stats.append([loss.mean(),cls_entropy,obj_entropy,center_unc,box_size_var,mean_obj_acc,masks])

       

    # # Log statistics
    # for key in sorted(stat_dict.keys()):
    #     log_string('eval mean %s: %f' % (key, stat_dict[key] /
    #                                      (float(batch_idx + 1))))

    #                             # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-' * 10, 'iou_thresh: %f' % (AP_IOU_THRESHOLDS[i]), '-' * 10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f' % (key, metrics_dict[key]))

    # mean_loss = stat_dict['loss'] / float(batch_idx + 1)
    # return mean_loss

    print("Obj ent acc mean ",np.array(obj_accs).mean())
    print("Cls ent acc mean ",np.array(cls_accs).mean())
    print("Size ent acc mean ",np.array(size_accs).mean())
    
    print("Obj Size ent acc mean ",np.array(obj_size_accs).mean())
    print("Obj Cls ent acc mean ",np.array(obj_cls_accs).mean())
    print("Cls size ent acc mean ",np.array(size_cls_accs).mean())
    print("Obj Cls Size ent acc mean ",np.array(obj_cls_size_accs).mean())

    return stats


def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    loss = evaluate_one_epoch()

def eval_uncertainties():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    return evaluate_one_epoch_with_uncertainties()
# if __name__ == '__main__':
#%%
# eval()
stats = eval_uncertainties()


# stats = np.array(stats)
# losses = stats.T[0]
# cls_entropies = stats.T[1][0]
# obj_entropies = stats.T[2][0]
# center_unces = stats.T[3][0]
# box_unces = stats.T[4][0]
# mean_obj_acc = stats.T[5][0]

# # %%
# import matplotlib.gridspec as gridspec

# fig = plt.gcf()
# fig.dpi =300
# gs1 = gridspec.GridSpec(3, 1)
# ax1 = fig.add_subplot(gs1[0])
# ax2 = fig.add_subplot(gs1[1])
# ax3 = fig.add_subplot(gs1[2])

# # ax2 = fig.add_subplot(gs1[1])

# ax1.set_title("Objectness entropy")
# ax1.plot(obj_entropies.squeeze(0),label="obj_entropy",color="b")

# # ax2.set_title("Mean objectness accuracy")
# # ax2.plot(mean_obj_acc,label="mean_obj_acc",color="g")

# # gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])

# # gs2 = gridspec.GridSpec(2, 1)
# # axes = []
# # for ss in gs2:
# #     ax = fig.add_subplot(ss)
# #     axes.append(ax)
    

#     #ax.set_xlabel("x-label", fontsize=12)
# ax2.plot(box_unces,label="box_size_var",color="r")
# ax2.set_title("Box size variance")
# ax2.set_xlabel("")
# # ax[0].set_ylabel("Avg box size variance")


# ax3.set_title("Classification entropy")
# ax3.plot(cls_entropies,label="cls_entropy",color="orange")

# # axes[2].set_title("Mean loss")
# # axes[2].plot(losses,label="loss",color = "black")

# gs1.tight_layout(fig, rect=[0.5, 0, 1, 1], h_pad=0.5)

# # top = gs
# # bottom = max(gs1.bottom, gs2.bottom)

# # gs1.update(top=top, bottom=bottom)
# # gs2.update(top=top, bottom=bottom)

# top = min(gs1.top, gs2.top)
# bottom = max(gs1.bottom, gs2.bottom)

# # gs1.tight_layout(fig, rect=[None, 0 + (bottom-gs1.bottom),
# #                             0.5, 1 - (gs1.top-top)])
# # gs2.tight_layout(fig, rect=[0.5, 0 + (bottom-gs2.bottom),
# #                             None, 1 - (gs2.top-top)],
#                 #  h_pad=0.5)
# # %%

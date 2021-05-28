# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" Evaluation routine for 3D object detection with SUN RGB-D and ScanNet.
"""
#!/usr/bin/env python

# %%
from models.ap_helper import parse_predictions_with_objectness_prob
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import APCalculator, parse_predictions_with_objectness_prob, parse_groundtruths,parse_predictions_with_custom_mask,parse_predictions
from utils.uncertainty_utils import map_zero_one, box_size_uncertainty, semantic_cls_uncertainty, objectness_uncertainty, center_uncertainty, apply_softmax, compute_objectness_accuracy, compute_iou_masks,compute_iou_masks_with_classification
from utils.binary_filter import UncertaintyFilter


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def create_class_dict():
    class_dict = {}
    for i in range(18):
        class_dict[i] = (0,0)
    return class_dict

FLAGS = AttrDict()
FLAGS.ap_iou_thresholds = '0.25,0.5'
FLAGS.batch_size = 1
# FLAGS.checkpoint_path = 'logs/dropoutsNotinBackbone/checkpoint208.tar'
FLAGS.checkpoint_path='logs/log_scannet120/checkpoint.tar'
# FLAGS.checkpoint_path='logs/log_scannet_dropout_0_1_200/checkpoint129.tar'
FLAGS.cluster_sampling = 'seed_fps'
FLAGS.conf_thresh = 0.5
FLAGS.dataset = 'scannet'
FLAGS.dump_dir = 'evals/test_colors'
FLAGS.faster_eval = True
FLAGS.model = 'votenet'
FLAGS.nms_iou = 0.25
FLAGS.no_height = False
FLAGS.num_point = 40000
FLAGS.num_target = 256
FLAGS.per_class_proposal = False
FLAGS.shuffle_dataset = False
FLAGS.use_3d_nms = True
FLAGS.use_cls_nms = True
FLAGS.use_color = False
FLAGS.use_old_type_nms = False
FLAGS.use_sunrgbd_v2 = False
FLAGS.vote_factor = 1
FLAGS.logdir = "logs/sanity_check_partialdropouts"
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DUMP_DIR = FLAGS.dump_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path
assert (CHECKPOINT_PATH is not None)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]
NUM_SAMPLES = 1
# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)
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


CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms': FLAGS.use_3d_nms, 'nms_iou': FLAGS.nms_iou,
    'use_old_type_nms': FLAGS.use_old_type_nms, 'cls_nms': FLAGS.use_cls_nms, 'per_class_proposal': FLAGS.per_class_proposal,
    'conf_thresh': FLAGS.conf_thresh, 'dataset_config':DATASET_CONFIG}
# ----------------------------------------------------------------------

# ------------------------------------------------------------------------- GLOBAL CONFIG END
def create_filters():
    objectness = UncertaintyFilter("Objectness Entropy Filter")
    classification = UncertaintyFilter("Classification Entropy Filter")
    box_size = UncertaintyFilter("Box Size Entropy Filter")
    objectness_and_classification = UncertaintyFilter("Objectness and Classification Entropy Filter")
    objectness_and_box_size = UncertaintyFilter("Objectness and Box Size Entropy Filter")
    box_size_and_classification = UncertaintyFilter("Box Size and Classification Entropy Filter")
    no_filter = UncertaintyFilter("All Ones Filter")
    models_filter = UncertaintyFilter("Model's Filter")
    models_voted_filter = UncertaintyFilter("Model's Voted Filter")

    model_and_objcls = UncertaintyFilter("Model and Obj & Cls Entropy Filter")
    model_and_obj = UncertaintyFilter("Model and Obj  Entropy Filter")
    model_and_cls = UncertaintyFilter("Model and Cls  Entropy Filter")
    return  {"objectness" : objectness,"classification": classification,
    "box_size":box_size,
    "objectness_and_classification":objectness_and_classification,
    "objectness_and_box_size":objectness_and_box_size,
    "box_size_and_classification":box_size_and_classification,
    "no_filter":no_filter,
    "models_filter" : models_filter,
    "models_voted_filter" : models_voted_filter,
    "model_and_objcls":model_and_objcls,
    "model_and_obj":model_and_obj,
    "model_and_cls":model_and_cls,

}



def evaluate_one_epoch_with_uncertainties():
    stat_dict = {}
    tb = SummaryWriter(FLAGS.logdir)

    ap_calculator_list_original = [APCalculator(iou_thresh, DATASET_CONFIG.class2type)
                          for iou_thresh in AP_IOU_THRESHOLDS]
    ap_calculator_list_custom = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval()  # set model to eval mode (for bn and dp)
    # net.enable_dropouts()
    
    stats = []
    class_acc_dict = {}
    for i in range(18):
            class_acc_dict[i] = [0,0]
    all_filters = create_filters()
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d' % (batch_idx))

        # 
        
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        loss = np.zeros([NUM_SAMPLES])
        with torch.no_grad():
            mc_samples = [net(inputs) for i in range(NUM_SAMPLES)]

        # Compute loss
        print("Sampling done")
        for idx, end_points in enumerate(mc_samples):
            for key in batch_data_label:
                assert (key not in end_points)
                end_points[key] = batch_data_label[key]
            local_loss, end_points = criterion(end_points, DATASET_CONFIG)
            loss[idx] = local_loss.cpu().detach().numpy().item()
        #     print(loss)
            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    if key not in stat_dict:
                        stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            # batch_pred_map_cls = parse_predictions_with_objectness_prob(end_points, CONFIG_DICT)
            # import copy
            # ep2 = copy.deepcopy(end_points)
            org_batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
            org_batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        # apply_softmax(mc_samples)

        # print("Softmaxed")

        # check = False
   
        # thresh = None
        # box_size_mask, box_size_var = box_size_uncertainty(mc_samples,thresh)
        # cls_entropy_mask, cls_entropy = semantic_cls_uncertainty(mc_samples,thresh)
        # obj_entropy_mask, obj_entropy = objectness_uncertainty(mc_samples,thresh)        
        
        # no_filter_mask = np.ones_like(box_size_mask)
        # models_masks = [sm["final_masks"][0] for sm in mc_samples]
        # models_mask = np.array(np.sum(np.array(models_masks),axis=0) > len(models_masks)*0.5,dtype=np.int)
        
    
        # all_filters["objectness"].set_mask(obj_entropy_mask)
        # all_filters["classification"].set_mask(cls_entropy_mask)
        # all_filters["box_size"].set_mask(box_size_mask)
        # all_filters["objectness_and_classification"].set_mask(np.logical_and(obj_entropy_mask,cls_entropy_mask)) 
        # all_filters["objectness_and_box_size"].set_mask(np.logical_and(obj_entropy_mask,box_size_mask)) 
        # all_filters["box_size_and_classification"].set_mask(np.logical_and(box_size_mask,cls_entropy_mask)) 
        # all_filters["no_filter"].set_mask(no_filter_mask) 
        # all_filters["models_filter"].set_mask(models_masks[0]) 
        # all_filters["models_voted_filter"].set_mask(models_mask) 
        # all_filters["model_and_cls"].set_mask(np.logical_and(models_mask,all_filters["classification"].fvector))  
        # all_filters["model_and_obj"].set_mask(np.logical_and(models_mask,all_filters["objectness"].fvector))  
        # all_filters["model_and_objcls"].set_mask(np.logical_and(models_mask,all_filters["objectness_and_classification"].fvector))  
        # cls_iou_masks,iou_masks = compute_iou_masks_with_classification(mc_samples,class_acc_dict)
#         print("Accuracy masks computed")
      
        
        # tf_logging_dict = {}
        # tf_logging_rejection = {}
        # for m in all_filters.keys():
        #     all_filters[m].update(mc_samples[0],iou_masks,cls_iou_masks)
        #     all_filters[m].log()
        #     obj_acc,cls_acc = all_filters[m].get_last_accs()
        #     tf_logging_dict[all_filters[m].name + "_obj_acc"] = obj_acc
        #     tf_logging_dict[all_filters[m].name + "_cls_acc"] = cls_acc
        #     tf_logging_rejection[all_filters[m].name + "_rejection"] = all_filters[m].num_rejected[-1]

     



        # tb.add_scalar("Box Size Variance",
        #               (box_size_var).mean(), batch_idx)
        # tb.add_scalar("Classification Entropy",
        #               (cls_entropy).mean(), batch_idx)
        # tb.add_scalar("Objectness Entropy",
        #               (obj_entropy).mean(), batch_idx)
        # tb.add_scalars("accs/", tf_logging_dict, batch_idx)
        # tb.add_scalars("rejections/", tf_logging_rejection, batch_idx)

        
 
        for ap_calculator in ap_calculator_list_original:
            ap_calculator.step(org_batch_pred_map_cls, org_batch_gt_map_cls)

        ep2["custom_mask"] = iou_masks[0]
        batch_pred_map_cls = parse_predictions_with_custom_mask(ep2, CONFIG_DICT)
        for ap_calculator in ap_calculator_list_custom:
            ap_calculator.step(batch_pred_map_cls, org_batch_gt_map_cls)
    

    
#     mean_obj_accs = []
#     mean_cls_accs = []
#     names = []
#     rejections = []
#     for f in all_filters.keys():   
#         mean_obj_accs.append(np.array(all_filters[f].obj_accs).mean())
#         mean_cls_accs.append(np.array(all_filters[f].cls_accs).mean())
#         names.append(f)
#         rejections.append(np.array(all_filters[f].num_rejected).sum())

#     fig, ax = plt.subplots()

#     y_pos = np.arange(len(names))
# ###############################

#     ax.barh(y_pos, np.array(mean_obj_accs), align='center')
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(names)
#     ax.invert_yaxis()  # labels read top-to-bottom
#     # ax.set_xlabel('Performance')
#     ax.set_title('"Mean Objectness Accuracies"')
#     plt.tight_layout()
#     tb.add_figure("Mean Objectness Accuracies",fig,0)

# ###############################
#     fig, ax = plt.subplots()

#     ax.barh(y_pos, np.array(mean_cls_accs), align='center')
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(names)
#     ax.invert_yaxis() 
#     ax.set_title('"Mean Classification Accuracies"')
#     plt.tight_layout()
#     tb.add_figure("Mean Classification Accuracies",fig,0)
# ###############################
#     fig, ax = plt.subplots()

#     ax.barh(y_pos, np.array(rejections), align='center')
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(names)
#     ax.invert_yaxis()  # labels read top-to-bottom
#     # ax.set_xlabel('Performance')
#     ax.set_title('"Rejections"')
#     plt.tight_layout()
#     tb.add_figure("Rejections",fig,0)

#     mean_obj_accs = []
#     mean_cls_accs = []
#     names = []
#     rejections = []
#     for f in all_filters.keys():   
#         mean_obj_accs.append(np.array(all_filters[f].obj_accs).mean())
#         mean_cls_accs.append(np.array(all_filters[f].cls_accs).mean())
#         names.append(f)
#         rejections.append(np.array(all_filters[f].num_rejected).sum())

#     fig, ax = plt.subplots()

#     y_pos = np.arange(len(names))
# ###############################

#     ax.barh(y_pos, np.array(mean_obj_accs), align='center')
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(names)
#     ax.invert_yaxis()  # labels read top-to-bottom
#     # ax.set_xlabel('Performance')
#     ax.set_title('"Mean Objectness Accuracies"')
#     plt.tight_layout()
#     tb.add_figure("Mean Objectness Accuracies",fig,0)

# ###############################
#     fig, ax = plt.subplots()

#     ax.barh(y_pos, np.array(mean_cls_accs), align='center')
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(names)
#     ax.invert_yaxis() 
#     ax.set_title('"Mean Classification Accuracies"')
#     plt.tight_layout()
#     tb.add_figure("Mean Classification Accuracies",fig,0)
# ###############################
#     fig, ax = plt.subplots()

#     ax.barh(y_pos, np.array(rejections), align='center')
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(names)
#     ax.invert_yaxis()  # labels read top-to-bottom
#     # ax.set_xlabel('Performance')
#     ax.set_title('"Rejections"')
#     plt.tight_layout()
#     tb.add_figure("Rejections",fig,0)

#     total_accs = []
#     mydict = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
#             'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
#             'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17, 'non-object':18} 
#     revDict = {}
#     for a in mydict.keys():
#         revDict[mydict[a]] = a
#     for f in all_filters.keys():
#         filt = all_filters[f]
#         accuracies = []
#         for i in filt.classification_class_dict.keys():
#             if np.sum(np.array(filt.classification_class_dict[i])) == 0:
#                 acc = -1
#             else:
#                 acc = filt.classification_class_dict[i][0]/(filt.classification_class_dict[i][0] + filt.classification_class_dict[i][1])
#             accuracies.append(acc)
#         total_accs.append(accuracies)
#     df = pd.DataFrame(total_accs,columns=list(mydict.keys()))
#     df.index = ([all_filters[a].name for a in all_filters.keys() ])
#     df.rename_axis("Filter")
#     df.to_csv(FLAGS.logdir + "/class_based_accuracies.csv")
#     tb.close()
    print("******************************ORIGINAL****************************************")
    for i, ap_calculator in enumerate(ap_calculator_list_original):
        print('-' * 10, 'iou_thresh: %f' % (AP_IOU_THRESHOLDS[i]), '-' * 10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f' % (key, metrics_dict[key]))

    print("******************************CUSTOM****************************************")
    for i, ap_calculator in enumerate(ap_calculator_list_custom):
        print('-' * 10, 'iou_thresh: %f' % (AP_IOU_THRESHOLDS[i]), '-' * 10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f' % (key, metrics_dict[key]))



    print("*********************CLASS ACCURACIES IN PROPOSALS*******************")
    mydict = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
        'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
        'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17} 
    revDict = {}
    for a in mydict.keys():
        revDict[mydict[a]] = a
    for c in revDict.keys():
        corr = class_acc_dict[c][0]
        wrong = class_acc_dict[c][1]
        if(corr + wrong == 0):
            acc = 0
        else:
            acc = corr/(corr + wrong )

        print(revDict[c],"Correct Match: ",corr,"Wrong Match: ",wrong,"Acc ",acc)
    return all_filters



def eval_uncertainties():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    return evaluate_one_epoch_with_uncertainties()


# if __name__ == '__main__':
all_filters = eval_uncertainties()
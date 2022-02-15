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

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath("/home/yildirir/workspace/votenet/README.md"))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
print(sys.path)
# from models.ap_helper import parse_predictions, parse_predictions_augmented, parse_predictions_ensemble_only_entropy
from ap_helper import APCalculator, parse_predictions_ensemble, parse_groundtruths,parse_predictions_augmented,aggregate_predictions,parse_predictions_ensemble_only_entropy
from uncertainty_utils import map_zero_one, box_size_uncertainty, semantic_cls_uncertainty, objectness_uncertainty, center_uncertainty, apply_softmax, compute_objectness_accuracy, compute_iou_masks,compute_iou_masks_with_classification
from dump_helper import dump_results_for_sanity_check,dump_results
import pc_util

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--selected_path', default=None, help='Selected path')
parser.add_argument('--unselected_path', default=None, help='Unselected path')
parser.add_argument('--custom_path', default=None, help='Custom data txt path [default: None]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--expected_ent', default=None, help='Use expected entropy instead of mutual information')

parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Point Number [default: 256]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--vote_factor', type=int, default=1, help='Number of votes generated from each seed [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use SUN RGB-D V2 box labels.')
parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')
parser.add_argument('--num_samples',type=int,default=5, help='Number of monte carlo sampling.')
parser.add_argument('--num_runs',type=int,default=1, help='Number of runs for the experiment.')
parser.add_argument('--num_batches',type=int,default=-1, help='Number of batches to process.')
parser.add_argument('--ratio', type=float, default=1, help='Fractional [default: 1]')
parser.add_argument('--dump_results', action='store_true', help='dump it or not')

parser.add_argument('--thresholds', type=float,nargs="+", default=[0.3],help='thresholds for rejecting objects while loading frames')
parser.add_argument('--chunk_size',type=int,default=120, help='Chunk size of a fraction.')
FLAGS = parser.parse_args()

if FLAGS.use_cls_nms:
    assert(FLAGS.use_3d_nms)


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DUMP_DIR = FLAGS.dump_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path
assert (CHECKPOINT_PATH is not None)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]
NUM_SAMPLES = FLAGS.num_samples
NUM_RUNS = FLAGS.num_runs
# Prepare DUMP_DIR
if FLAGS.dump_results:
    if not os.path.exists(DUMP_DIR):
        os.mkdir(DUMP_DIR)
    DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
    DUMP_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    if FLAGS.dump_results:
        DUMP_FOUT.write(out_str + '\n')
        DUMP_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    np.random.seed(1)



angles = [np.pi/2, np.pi,3*np.pi/2]
# angles = []
# rotations = [np.eye(3)] + [pc_util.rotx(a) for a in angles] + [pc_util.roty(a) for a in angles] + [pc_util.rotz(a) for a in angles]
rotations = [np.eye(3)] +  [pc_util.rotz(a) for a in angles]
# rotations = [pc_util.rotz(a) for a in angles]
print(rotations)
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

    # TEST_DATASETS = [ScannetDetectionDataset('val',
    #                                        num_points=NUM_POINT,
    #                                        augment=False,
    #                                        use_color=FLAGS.use_color,
    #                                        use_height=(not FLAGS.no_height),rot=rot) for rot in rotations]
    if FLAGS.custom_path is None:
        TEST_DATASET = ScannetDetectionDataset('val',
                                            num_points=NUM_POINT,
                                            augment=False,
                                            use_color=FLAGS.use_color,
                                            use_height=(not FLAGS.no_height))
        
    else:
        TEST_DATASET = ScannetDetectionDataset('fractional_train',
                                            num_points=NUM_POINT,
                                            augment=False,
                                            use_color=FLAGS.use_color,
                                            use_height=(not FLAGS.no_height),custom_path=FLAGS.custom_path)
elif FLAGS.dataset == 'scannet_frames':

    # sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    # sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    # from scannet_frames_dataset import ScannetDetectionFramesDataset, MAX_NUM_OBJ
    # from model_util_scannet import ScannetDatasetConfig
    # DATASET_CONFIG = ScannetDatasetConfig()
    # data_setting = {
    #     "dataset_path":"/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/",
    #     "train_source":"/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/train_frames.txt",
    #     "eval_source":"/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/val_frames.txt",
    #     "frames_path": "/home/yildirir/workspace/kerem/TorchSSC/DATA/scannet_frames_25k/"
    # }
    # TEST_DATASET = ScannetDetectionFramesDataset(data_setting,split_set="val",num_points=NUM_POINT,use_color=False,use_height=True,augment=False)
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_frames_dataset import ScannetDetectionFramesDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    data_setting = {
        "dataset_path":"/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/",
        "train_source":"/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/train_frames.txt",
        "eval_source":"/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/val_frames.txt",
        "frames_path": "/home/yildirir/workspace/kerem/TorchSSC/DATA/scannet_frames_25k/"
    }
    
    TEST_DATASETS = [ScannetDetectionFramesDataset(data_setting,split_set="val",num_points=NUM_POINT,use_color=False,use_height=True,augment=False,thresh=t) for t in FLAGS.thresholds]
         
elif FLAGS.dataset == 'scannet_single_sanity':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    
    TEST_DATASET = ScannetDetectionDataset('single_eval', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))

else:
    print('Unknown dataset %s. Exiting...' % (FLAGS.dataset))
    exit(-1)
# print(len(TEST_DATASET))
TEST_DATALOADERS =[ DataLoader(TEST_DATASET,
                             batch_size=BATCH_SIZE,
                             shuffle=FLAGS.shuffle_dataset,
                             num_workers=1,
                             worker_init_fn=my_worker_init_fn) for TEST_DATASET in TEST_DATASETS]

# TEST_DATALOADERS = [DataLoader(TEST_DATASETS[i],
#                              batch_size=BATCH_SIZE,
#                              shuffle=FLAGS.shuffle_dataset,
#                              num_workers=4,
#                              worker_init_fn=my_worker_init_fn) for i in range(len(rotations))]

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
else:
    log_string("INVALID CHECKPOINT")
CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms': FLAGS.use_3d_nms, 'nms_iou': FLAGS.nms_iou,
    'use_old_type_nms': FLAGS.use_old_type_nms, 'cls_nms': FLAGS.use_cls_nms, 'per_class_proposal': FLAGS.per_class_proposal,
    'conf_thresh': FLAGS.conf_thresh, 'dataset_config':DATASET_CONFIG}


print(CONFIG_DICT)
print(FLAGS)

# -------------------
# ----------------------------------------------------------------------

# ------------------------------------------------------------------------- GLOBAL CONFIG END


#for single scene make it for batch later
def prep_augmented_pcds(inputs,rotations):
    
    num_batches = inputs["point_clouds"].shape[0]
    aug_pcds = [None]*len(rotations)
    for idx,rotmat in enumerate(rotations):
        aug_pcds[idx] = []
        for b in range(num_batches):
            pcd = inputs["point_clouds"][b].clone()
            xyz = pcd[:,0:3].contiguous()
            rotated,_ = pc_util.rotate_point_cloud(xyz.cpu().numpy(),rotation_matrix=rotmat)
            pcd[:,0:3] = torch.Tensor(rotated).to(device)
            aug_pcds[idx].append(pcd)
        aug_pcds[idx] = {"point_clouds":torch.stack(aug_pcds[idx])}
    
    return aug_pcds




def evaluate_one_epoch_with_uncertainties_mc():
    stat_dict = {}
    # tb = SummaryWriter(FLAGS.logdir)
    methods = ["Native","objectness","classification", #TODO: add box size
        "obj_and_cls"]
    # methods = ["Native"]
    # methods = ["obj_and_cls","Native"]
    met_dict = {}
    unc_dict = {}
    for m in methods:
        met_dict[m] =  [APCalculator(iou_thresh, DATASET_CONFIG.class2type)
                          for iou_thresh in AP_IOU_THRESHOLDS]
        unc_dict[m] = [0,0]
    # ap_calculator_list_original = [APCalculator(iou_thresh, DATASET_CONFIG.class2type)
    #                       for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval()  # set model sem_cls_probs[i,j,ii]to eval mode (for bn and dp)
    net.enable_dropouts()
    
    for TEST_DATALOADER in TEST_DATALOADERS: 

        print("------------------------------------------{}-------------------------------------------".format(TEST_DATALOADER.dataset.thresh))
        for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
            if batch_idx % 10 == 0:
                print('Eval batch: %d' % (batch_idx))
            if batch_idx < 2:
                continue
            if batch_idx == FLAGS.num_batches:
                
                break
            for key in batch_data_label:
                if key != 'name':
                    batch_data_label[key] = batch_data_label[key].to(device)

            # Forward pass
            inputs = {'point_clouds': batch_data_label['point_clouds'],'name':batch_data_label["name"]}
            loss = np.zeros([NUM_SAMPLES])
            with torch.no_grad():
                mc_samples = [net(inputs) for i in range(NUM_SAMPLES)]

            # print("Batch ",batch_idx, " samples done")
            #NEED TO UNDO
            # dump_results_for_sanity_check(batch_data_label,FLAGS.dump_dir,DATASET_CONFIG)
        #     # Compute loss
            for idx, end_points in enumerate(mc_samples):
                for key in batch_data_label:
                    assert (key not in end_points)
                    end_points[key] = batch_data_label[key]
                local_loss, end_points = criterion(end_points, DATASET_CONFIG)


        #     # center_uncertainty(mc_samples)        
        #     #This guy has len(methods) elements
            import copy
            # print("Predictions parsing")
            batch_pred_map_cls =[parse_predictions_ensemble(copy.deepcopy(mc_samples), CONFIG_DICT,m,FLAGS.expected_ent) for m in methods]
            # print("Predictions computed")
            bsize = FLAGS.batch_size
            
            for idx,m in enumerate(methods):
                num_proposals = np.array([np.sum(len(batch_pred_map_cls[idx][i]) for i in range(bsize))]) # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
                obj_and_cls_uncertainties = np.array([np.sum([ (a[2]) for a in batch_pred_map_cls[idx][i]]) for i in range(bsize)]) # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
                # print("{} {} {}".format(m,np.sum(num_proposals),obj_and_cls_uncertainties.sum()))
                unc_dict[m] += np.array([np.sum(num_proposals),obj_and_cls_uncertainties.sum()])
            org_batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
            

            for idx,m in enumerate(methods):
                for ap_calculator in met_dict[m]:
                    ap_calculator.step(batch_pred_map_cls[idx], org_batch_gt_map_cls)


        
        for k in unc_dict.keys():
            summed_elem = unc_dict[k]
            # print("{} {}".format(k,summed_elem/len(TEST_DATASET)))
        for idx,m in enumerate(methods):
            print("|",m,"|","| ")
            for ap_calculator in met_dict[m]:
                print('|', 'iou_thresh | %f  ' % (ap_calculator.ap_iou_thresh), ' | ')
                metrics_dict = ap_calculator.compute_metrics()
                for key in metrics_dict:
                    if key == "mAP" or key == "AR":
                        log_string(' | %s  | %f | ' % (key,metrics_dict[key]))
        
        print("------------------------------------------END OF {}-------------------------------------------".format(TEST_DATALOADER.dataset.thresh))
def compute_uncertainties_mc():
    stat_dict = {}
    # tb = SummaryWriter(FLAGS.logdir)
    methods = ["obj_and_cls"]
    # methods = ["Native"]
    # methods = ["obj_and_cls","Native"]
    met_dict = {}
    unc_dict = {}
    for m in methods:
        met_dict[m] =  [APCalculator(iou_thresh, DATASET_CONFIG.class2type)
                          for iou_thresh in AP_IOU_THRESHOLDS]
        unc_dict[m] = 0
    # ap_calculator_list_original = [APCalculator(iou_thresh, DATASET_CONFIG.class2type)
    #                       for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval()  # set model sem_cls_probs[i,j,ii]to eval mode (for bn and dp)
    net.enable_dropouts()
    
    entropies = []
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        print("INSIDE")
        if batch_idx % 10 == 0:
            print('Eval batch: %d' % (batch_idx))

        if batch_idx == FLAGS.num_batches:
            
            break
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        loss = np.zeros([NUM_SAMPLES])
        with torch.no_grad():
            mc_samples = [net(inputs) for i in range(NUM_SAMPLES)]

        for idx, end_points in enumerate(mc_samples):
            for key in batch_data_label:
                assert (key not in end_points)
                end_points[key] = batch_data_label[key]
        
        dump_results_for_sanity_check(end_points,FLAGS.DUMP_DIR, DATASET_CONFIG )
        """
        #NEED TO UNDO
        if FLAGS.dump_results:
            if not os.path.exists(FLAGS.dump_dir + "_batch_{}".format(batch_idx)):
                os.makedirs(FLAGS.dump_dir + "_batch_{}".format(batch_idx))
        # dump_results_for_sanity_check(batch_data_label,FLAGS.dump_dir + "_batch_{}".format(batch_idx),DATASET_CONFIG)
                scan_name_path = os.path.join(FLAGS.dump_dir + "_batch_{}".format(batch_idx),"scan_names.txt")

            
    #     # Compute loss
          #This guy has len(methods) elements
        unc =parse_predictions_ensemble_only_entropy(mc_samples, CONFIG_DICT,"obj_and_cls")
       
       # unc_sum = unc.sum()
        for i in unc:
            entropies.append(i)
    #0.1 of original is 120
        if FLAGS.dump_results:
            entropies_in_scenes = np.array(entropies)[batch_data_label["scan_idx"].cpu()]
            with open(scan_name_path,"w") as f:
                files = np.array(TEST_DATASET.scan_names)[batch_data_label["scan_idx"].cpu()]
                for idx,l in enumerate(files):
                    f.write(l + " " + str(entropies_in_scenes[idx]) + "\n" )

    chunk_size = FLAGS.chunk_size
    entropies = np.array(entropies)
    dicty = {}
    for idx,line in enumerate(TEST_DATASET.scan_names[:(batch_idx)*BATCH_SIZE]):
        name = line[:line.find("_")] 
        if name in dicty.keys():
            dicty[name][0].append(line[:])
            dicty[name][1].append(entropies[idx])
        else:
            dicty[name] = [[line[:]],[entropies[idx]]]
    
    scenes = list(dicty.keys())
    entropies = np.array([np.mean(ent[1][1]) for ent in dicty.items()])
    entropy_sorted_inds = np.argsort(entropies)[::-1]
    sorted_entropies = entropies[entropy_sorted_inds]
    sorted_scenes = np.array(scenes)[entropy_sorted_inds]
    sorted_scan_names = []
    for s in sorted_scenes:
        elems = dicty[s][0]
        for e in elems:
            sorted_scan_names.append(e)

    sorted_scan_names = np.array(sorted_scan_names)
    selected_scan_names =sorted_scan_names[:chunk_size]
    # selected_scan_entropies = sorted_entropies[:chunk_size]
    unselected_scan_names = sorted_scan_names[chunk_size:]
    unselected_scan_entropies = sorted_entropies[chunk_size:]
    with open(FLAGS.selected_path,"a+") as f:
        for idx,n in enumerate(selected_scan_names):
            f.write(str(n) +  "\n")    
    with open(FLAGS.unselected_path,"w+") as f:
        for idx,n in enumerate(unselected_scan_names):
            f.write(str(n) + "\n")    
        """
        
def evaluate_one_epoch_with_uncertainties_augmented():
    stat_dict = {}
    global rotations
    # tb = SummaryWriter(FLAGS.logdir)

    

    # angles = [np.pi]

    # methods = ["Native","objectness","classification", #TODO: add box size
    #     "obj_and_cls"]
    methods = ["Native"]
    # methods = ["obj_and_cls","Native"]
    met_dict = {}
    # for m in methods:
    #     met_dict[m] =  [APCalculator(iou_thresh, DATASET_CONFIG.class2type)
    #                       for iou_thresh in AP_IOU_THRESHOLDS]
    ap_calculator_list_original = [APCalculator(iou_thresh, DATASET_CONFIG.class2type)
                          for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval()  # set model sem_cls_probs[i,j,ii]to eval mode (for bn and dp)
    # net.enable_dropouts()
    
    
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d' % (batch_idx))

        if batch_idx == FLAGS.num_batches:
            
            break
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass

        
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        
        aug_pcds = prep_augmented_pcds(inputs,rotations)
        loss = np.zeros([NUM_SAMPLES])
        with torch.no_grad():

            mc_samples = [net(pcd) for pcd in aug_pcds]


        #NEED TO UNDO
        # dump_results_for_sanity_check(batch_data_label,FLAGS.dump_dir,DATASET_CONFIG)
    #     # Compute loss
        for idx, end_points in enumerate(mc_samples):
            for key in batch_data_label:
                assert (key not in end_points)
                end_points[key] = batch_data_label[key]
            local_loss, end_points = criterion(end_points, DATASET_CONFIG)


    #     # center_uncertainty(mc_samples)        
    #     #This guy has len(methods) elements
        import copy
        # batch_pred_map_cls =[parse_predictions_augmented(copy.deepcopy(mc_samples), CONFIG_DICT,m,rotations) for m in methods]
        
        # batch_pred_map_cls =[parse_predictions_ensemble([copy.deepcopy(mc_samples[3])], CONFIG_DICT,m) for m in methods]
        batch_pred_map_cls_multiple = [parse_predictions(m,CONFIG_DICT,noflip=False) for m in mc_samples]
        #need to undo the rotations, but is the flip have any effect?
        
        batch_pred_map_cls = aggregate_predictions(batch_pred_map_cls_multiple,rotations,dump_dir=FLAGS.DUMP_DIR)
        # batch_pred_map_cls = batch_pred_map_cls_multiple[0]
        end_points = mc_samples[0]
        end_points["batch_pred_map_cls"] = batch_pred_map_cls
        # batch_pred_map_cls = parse_predictions(mc_samples[0],CONFIG_DICT,noflip=False)
        org_batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        
        
        for ap_calculator in ap_calculator_list_original:
            ap_calculator.step(batch_pred_map_cls, org_batch_gt_map_cls)

    
        dump_results(end_points,FLAGS.DUMP_DIR, DATASET_CONFIG )
        
    for idx,m in enumerate(methods):
        print("|",m,"|","| ")
        for ap_calculator in ap_calculator_list_original:
            print('|', 'iou_thresh | %f  ' % (ap_calculator.ap_iou_thresh), ' | ')
            metrics_dict = ap_calculator.compute_metrics()
            for key in metrics_dict:
                if key == "mAP" or key == "AR":
                    log_string(' | %s  | %f | ' % (key,metrics_dict[key]))

def evaluate_one_epoch_with_rotation():
    stat_dict = {}
    # tb = SummaryWriter(FLAGS.logdir)

    
    # angles = [np.pi/2,np.pi]
    # # angles = [np.pi]
    # rotations = [np.eye(3)] + [pc_util.rotx(a) for a in angles] + [pc_util.roty(a) for a in angles] + [pc_util.rotz(a) for a in angles]

    methods = ["Native"]
    # methods = ["obj_and_cls","Native"]
    
    
    met_dict = {}
    for m in methods:
        met_dict[m] =  [APCalculator(iou_thresh, DATASET_CONFIG.class2type)
                          for iou_thresh in AP_IOU_THRESHOLDS]
    # ap_calculator_list_original = [APCalculator(iou_thresh, DATASET_CONFIG.class2type)
    #                       for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval()  # set model sem_cls_probs[i,j,ii]to eval mode (for bn and dp)
    # net.enable_dropouts()
    
    for didx,TEST_DATALOADER in enumerate(TEST_DATALOADERS):
        print("Evaluating dataloader ",didx)
        for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
            if batch_idx % 10 == 0:
                print('Eval batch: %d' % (batch_idx))

            if batch_idx == FLAGS.num_batches:
                
                break
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)

            # Forward pass

            
            inputs = {'point_clouds': batch_data_label['point_clouds']}
            
            # loss = np.zeros([NUM_SAMPLES])
            with torch.no_grad():

                end_points = net(inputs) 


            #NEED TO UNDO
            # dump_results_for_sanity_check(batch_data_label,FLAGS.dump_dir,DATASET_CONFIG)
        #     # Compute loss
            # for idx, end_points in enumerate(mc_samples):
            # dump_results_for_sanity_check(batch_data_label,FLAGS.DUMP_DIR + str(didx), DATASET_CONFIG)
            for key in batch_data_label:
                assert (key not in end_points)
                end_points[key] = batch_data_label[key]
            local_loss, end_points = criterion(end_points, DATASET_CONFIG)


        # #     # center_uncertainty(mc_samples)        
        # #     #This guy has len(methods) elements
        #     import copy
            batch_pred_map_cls = [parse_predictions(end_points,CONFIG_DICT) for m in methods]
        #     # batch_pred_map_cls =[parse_predictions_ensemble(copy.deepcopy(mc_samples), CONFIG_DICT,m) for m in methods]
        #     #gotta rotate
            org_batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)

            # for idx,m in enumerate(methods):
            #     for ap_calculator in met_dict[m]:
            #         ap_calculator.step(batch_pred_map_cls[idx], org_batch_gt_map_cls)


        
    
        # for idx,m in enumerate(methods):
        #     print("|",m,"|","| ")
        #     for ap_calculator in met_dict[m]:
        #         print('|', 'iou_thresh | %f  ' % (ap_calculator.ap_iou_thresh), ' | ')
        #         metrics_dict = ap_calculator.compute_metrics()
        #         for key in metrics_dict:
        #             if key == "mAP" or key == "AR":
        #                 log_string(' | %s  | %f | ' % (key,metrics_dict[key]))
        
        
        # dump_results(end_points,FLAGS.DUMP_DIR + str(didx), DATASET_CONFIG)

        # if didx > 1:
        #     break



def eval_uncertainties():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    
    # return evaluate_one_epoch_with_uncertainties_augmented()
    if FLAGS.custom_path is None:
        return evaluate_one_epoch_with_uncertainties_mc()
    else:
        return compute_uncertainties_mc()

    # return evaluate_one_epoch_with_rotation()

# if __name__ == '__main__':
np.random.seed(1)
for i in range(NUM_RUNS):
    
    eval_uncertainties()

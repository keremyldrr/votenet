#!/usr/bin/env python

#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
BASE_DIR =os.path.dirname(os.path.abspath("models"))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import APCalculator, parse_predictions, parse_groundtruths ,softmax


# ## Setting up initial parameters and paths

# In[2]:


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

FLAGS = AttrDict()
FLAGS.ap_iou_thresholds='0.25,0.5'
FLAGS.batch_size=1
FLAGS.checkpoint_path='log_scannet_dropout_0_3_200/checkpoint129.tar'
FLAGS.cluster_sampling='seed_fps'
FLAGS.conf_thresh=0.05
FLAGS.dataset='scannet'
FLAGS.dump_dir='evals/eval_scannet_dropout_0_3_129_jupy'
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

if FLAGS.use_cls_nms:
    assert(FLAGS.use_3d_nms)

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DUMP_DIR = FLAGS.dump_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path
assert(CHECKPOINT_PATH is not None)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]

# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    DUMP_FOUT.write(out_str+'\n')
    DUMP_FOUT.flush()
    print(out_str)

# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)
print(len(TEST_DATASET))
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
    shuffle=FLAGS.shuffle_dataset, num_workers=4, worker_init_fn=my_worker_init_fn)

# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

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
    log_string("Loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, epoch))

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms': FLAGS.use_3d_nms, 'nms_iou': FLAGS.nms_iou,
    'use_old_type_nms': FLAGS.use_old_type_nms, 'cls_nms': FLAGS.use_cls_nms, 'per_class_proposal': FLAGS.per_class_proposal,
    'conf_thresh': FLAGS.conf_thresh, 'dataset_config':DATASET_CONFIG}
# ------------------------------------------------------------------------- GLOBAL CONFIG END
stat_dict = {}
ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) for iou_thresh in AP_IOU_THRESHOLDS]
net.eval()
net.pnet.drop1.train()
net.pnet.drop2.train()
net.pnet.drop3.train()


# In[ ]:





# In[3]:


def apply_softmax(samples):
    for e in samples:
        e["objectness_scores"] = softmax(e["objectness_scores"].cpu().detach().numpy())
        e["sem_cls_scores"] = softmax(e["sem_cls_scores"].cpu().detach().numpy())[0]


# # Epistemic uncertainty for classification on MLP output

# In[4]:


def semantic_cls_uncertainty(samples):
    mc_cls = np.array([e["sem_cls_scores"] for e in samples])
    expected_p = np.mean(mc_cls, axis=0)
    predictive_entropy = -np.sum(expected_p *np.log(expected_p), axis=-1)
    MC_entropy = np.sum(mc_cls * np.log(mc_cls),axis=-1)
    expected_entropy = -np.mean(MC_entropy, axis=0)
    mi = predictive_entropy - expected_entropy
    return mi.sum()


# # Epistemic Uncertainty on objectness score

# In[5]:


def objectness_uncertainty(samples):
    mc_objs = np.array([e["objectness_scores"] for e in samples])
    expected_p_obj = np.mean(mc_objs, axis=0)
    predictive_entropy_obj = -np.sum(expected_p_obj *np.log(expected_p_obj), axis=-1)
    MC_entropy_obj = np.sum(mc_objs * np.log(mc_objs),axis=-1)
    expected_entropy_obj = -np.mean(MC_entropy_obj, axis=0)
    mi_obj = predictive_entropy_obj - expected_entropy_obj
    return mi_obj.sum()


# # Epistemic uncertainty of center proposals

# In[6]:


def center_uncertainty(samples):
    expected_centers = torch.zeros([256,3]).unsqueeze(0)
    centers = []
    for e in samples:
        centers.append(e["center"])
    center_variance = np.var(np.array([c.squeeze(0).cpu().detach().numpy() for c in centers]),axis=0)
    center_mean =np.mean(np.array([c.squeeze(0).cpu().detach().numpy() for c in centers]),axis=0)
    return center_variance.sum()


# In[11]:


stat_dict = {}
ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type)     for iou_thresh in AP_IOU_THRESHOLDS]
# net.eval() # set model to eval mode (for bn and dp)
stats = []
for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
    if batch_idx % 10 == 0:
        print('Eval batch: %d'%(batch_idx))
    for key in batch_data_label:
        batch_data_label[key] = batch_data_label[key].to(device)

    # Forward pass
    inputs = {'point_clouds': batch_data_label['point_clouds']}
    with torch.no_grad():
        MC_end_points = [net(inputs) for i in range(1)]
    for key in batch_data_label:
        for end_points in MC_end_points:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
    loss = 0
    for end_points in MC_end_points:
        loss, end_points = criterion(end_points, DATASET_CONFIG)
        loss += loss/len(MC_end_points)
#     apply_softmax(MC_end_points)
#     cls_entropy = semantic_cls_uncertainty(MC_end_points)
#     obj_entropy = objectness_uncertainty(MC_end_points)
#     center_unc = center_uncertainty(MC_end_points)
#     stats.append([loss.cpu().detach().numpy().item(),cls_entropy,obj_entropy,center_unc])
#     print(loss)
    # Accumulate statistics and print out
    for key in end_points:
        if 'loss' in key or 'acc' in key or 'ratio' in key:
            if key not in stat_dict: stat_dict[key] = 0
            stat_dict[key] += end_points[key].item()

    batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
    batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
    for ap_calculator in ap_calculator_list:

        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
    ap_calculator_list[0].compute_metrics()
    break
#     if batch_idx == 20:
#         break


# In[8]:


# batch_gt_map_cls


# In[ ]:





# In[ ]:





# In[ ]:


# stats_arr = np.array(stats)
# import matplotlib.pyplot as plt


# In[ ]:


# losses = stats_arr.T[0]
# cls_entropies = stats_arr.T[1]
# obj_entropies = stats_arr.T[2]
# center_unces = stats_arr.T[3]


# In[ ]:





# In[ ]:


# plt.figure(dpi=300)
# plt.plot(losses,label="loss")
# plt.plot(cls_entropies,label="cls_entropy")
# plt.plot(obj_entropies,label="obj_entropy")
# plt.plot(center_unces,label="center_uncertainty")
# plt.legend()
# plt.title("Dropout p = 0.3")

# plt.savefig('0.3_uncertainty_vs_loss_smaller.png', dpi=300)
# plt.show()


# In[ ]:



# def evaluate_one_epoch():
#     stat_dict = {}
#     ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
#         for iou_thresh in AP_IOU_THRESHOLDS]
#     net.eval() # set model to eval mode (for bn and dp)
#     for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
#         if batch_idx % 10 == 0:
#             print('Eval batch: %d'%(batch_idx))
#         for key in batch_data_label:
#             batch_data_label[key] = batch_data_label[key].to(device)

#         # Forward pass
#         inputs = {'point_clouds': batch_data_label['point_clouds']}
#         with torch.no_grad():
#             end_points = net(inputs)

#         # Compute loss
#         for key in batch_data_label:
#             assert(key not in end_points)
#             end_points[key] = batch_data_label[key]
#         loss, end_points = criterion(end_points, DATASET_CONFIG)

#         # Accumulate statistics and print out
#         for key in end_points:
#             if 'loss' in key or 'acc' in key or 'ratio' in key:
#                 if key not in stat_dict: stat_dict[key] = 0
#                 stat_dict[key] += end_points[key].item()

#         batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
#         batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
#         for ap_calculator in ap_calculator_list:
#             ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

#         # Dump evaluation results for visualization
# #         if batch_idx == 0:
# #             MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

#     # Log statistics
#     for key in sorted(stat_dict.keys()):
#         log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

#     # Evaluate average precision
#     for i, ap_calculator in enumerate(ap_calculator_list):
#         print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
#         metrics_dict = ap_calculator.compute_metrics()
#         for key in metrics_dict:
#             log_string('eval %s: %f'%(key, metrics_dict[key]))

#     mean_loss = stat_dict['loss']/float(batch_idx+1)
#     return mean_loss

# def eval():
#     log_string(str(datetime.now()))
#     # Reset numpy seed.
#     # REF: https://github.com/pytorch/pytorch/issues/5059
#     np.random.seed()
#     loss = evaluate_one_epoch()


# In[ ]:


# eval()


# In[ ]:


# import os
# import sys
# import numpy as np
# import argparse
# import importlib
# import time

# class AttrDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self

# FLAGS = AttrDict()
# FLAGS.ap_iou_thresholds='0.25,0.5'
# FLAGS.batch_size=1
# FLAGS.checkpoint_path='log_scannet_dropout_0_3_200/checkpoint129.tar'
# FLAGS.cluster_sampling='seed_fps'
# FLAGS.conf_thresh=0.05
# FLAGS.dataset='scannet'
# FLAGS.dump_dir='evals/eval_scannet_dropout_0_3_129_jupy'
# FLAGS.faster_eval=False
# FLAGS.model='votenet'
# FLAGS.nms_iou=0.25
# FLAGS.no_height=False
# FLAGS.num_point=40000
# FLAGS.num_target=256
# FLAGS.per_class_proposal=True
# FLAGS.shuffle_dataset=False
# FLAGS.use_3d_nms=True
# FLAGS.use_cls_nms=True
# FLAGS.use_color=False
# FLAGS.use_old_type_nms=False
# FLAGS.use_sunrgbd_v2=False
# FLAGS.vote_factor=1

# import torch
# import torch.nn as nn
# import torch.optim as optim

# BASE_DIR = os.path.dirname(os.path.abspath("models"))
# ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
# from pc_util import random_sampling, read_ply
# from ap_helper import parse_predictions

# def preprocess_point_cloud(point_cloud):
#     ''' Prepare the numpy point cloud (N,3) for forward pass '''
#     point_cloud = point_cloud[:,0:3] # do not use color for now
#     floor_height = np.percentile(point_cloud[:,2],0.99)
#     height = point_cloud[:,2] - floor_height
#     point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
#     point_cloud = random_sampling(point_cloud, FLAGS.num_point)
#     pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
#     return pc

# if __name__=='__main__':

#     # Set file paths and dataset config
#     demo_dir = os.path.join(BASE_DIR, 'demo_files')
#     if FLAGS.dataset == 'sunrgbd':
#         sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
#         from sunrgbd_detection_dataset import DC # dataset config
#         checkpoint_path = FLAGS.checkpoint_path# os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
#         pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')
#     elif FLAGS.dataset == 'scannet':
#         sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
#         from scannet_detection_dataset import DC # dataset config
#         checkpoint_path = FLAGS.checkpoint_path# os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
#         pc_path = os.path.join(demo_dir, 'input_pc_scannet.ply')
#     else:
#         print('Unkown dataset %s. Exiting.'%(DATASET))
#         exit(-1)

#     eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
#         'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
#         'conf_thresh': 0.5, 'dataset_config': DC}

#     # Init the model and optimzier
#     MODEL = importlib.import_module('votenet') # import network module
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net = MODEL.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
#         sampling='seed_fps', num_class=DC.num_class,
#         num_heading_bin=DC.num_heading_bin,
#         num_size_cluster=DC.num_size_cluster,
#         mean_size_arr=DC.mean_size_arr).to(device)
#     print('Constructed model.')

#     # Load checkpoint
#     optimizer = optim.Adam(net.parameters(), lr=0.001)
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))

#     # Load and preprocess input point cloud
#     net.eval() # set model to eval mode (for bn and dp)
#     net.pnet.drop1.train()
#     net.pnet.drop2.train()
#     net.pnet.drop3.train()
#     point_cloud = read_ply(pc_path)
#     pc = preprocess_point_cloud(point_cloud)
#     print('Loaded point cloud data: %s'%(pc_path))

#     # Model inference
#     inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
#     tic = time.time()
#     with torch.no_grad():
#         end_points = net(inputs)
#     toc = time.time()
#     print('Inference time: %f'%(toc-tic))
#     end_points['point_clouds'] = inputs['point_clouds']
#     pred_map_cls = parse_predictions(end_points, eval_config_dict)
#     print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))

#     dump_dir = os.path.join(demo_dir, '%s_results'%(FLAGS.dataset))
#     if not os.path.exists(dump_dir): os.mkdir(dump_dir)
# #     MODEL.dump_results(end_points, dump_dir, DC, True)
#     print('Dumped detection results to folder %s'%(dump_dir))

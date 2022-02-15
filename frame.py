 
import numpy as np
import trimesh
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath("/home/yildirir/workspace/votenet/README.md"))
ROOT_DIR = BASE_DIR

sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
from scannet_frames_dataset import ScannetDetectionFramesDataset 
from scannet.scannet_detection_dataset import ScannetDetectionDataset
from model_util_scannet import ScannetDatasetConfig
from models.dump_helper import dump_results_for_sanity_check
import torch


from model_util_scannet import ScannetDatasetConfig

DC = ScannetDatasetConfig()



NUM_POINT = 20000


data_setting = {
    "dataset_path":"/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/",
    "train_source":"/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/train_frames.txt",
    "eval_source":"/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/val_frames.txt",
    "frames_path": "/home/yildirir/workspace/kerem/TorchSSC/DATA/scannet_frames_25k/"

}

threshes = [0.3,0.5,0.7, 0.8]

for t in threshes:
    TRAIN_DATASET = ScannetDetectionFramesDataset(data_setting,split_set="train",num_points=NUM_POINT,use_color=False,use_height=True,augment=False,thresh=t)
    print(len(TRAIN_DATASET))
    
# TRAIN_DATASET_WHOLE= ScannetDetectionDataset('train', num_points=NUM_POINT,augment=True,use_color=False, use_height=True)
 



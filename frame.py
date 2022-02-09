 
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
TRAIN_DATASET = ScannetDetectionFramesDataset(data_setting,split_set="train",num_points=NUM_POINT,use_color=False,use_height=True,augment=False)
    
TRAIN_DATASET_WHOLE= ScannetDetectionDataset('train', num_points=NUM_POINT,augment=True,use_color=False, use_height=True)
 
elem = TRAIN_DATASET[90]
scene_elem = TRAIN_DATASET_WHOLE[170]
# dump_results_for_sanity_check(elem,dump_dir=".",config=DC)
print(elem.keys())
for idx,(val1,val2) in enumerate(zip(list(elem.values()),list(scene_elem.values()))):
    # print(val1.shape,val2.shape)
    print(list(elem.keys())[idx],np.unique(val1),np.unique(val2)) 

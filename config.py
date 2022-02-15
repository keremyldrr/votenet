#!/usr/bin/env python3

from easydict import EasyDict as edict
import torch

C =edict()
C.MODEL = 'votenet'           #Model file name [default: votenet]')
C.DATASET = 'scannet_frames'         #Dataset name. sunrgbd or scannet. [default: sunrgbd]')
C.CHECKPOINT_PATH  = None     #Model checkpoint path [default: None]')
C.CUSTOM_PATH =None           #Custom data txt path [default: None]')
C.LOG_DIR = 'log' #Dump dir to save model checkpoint [default: log]')
C.DUMP_DIR = None             #Dump dir to save sample outputs [default: None]')
C.NUM_POINTS  = 20000           #Point Number [default: 20000]')
C.NO_HEIGHT = False
C.NUM_TARGET =256             #Proposal number [default: 256]')
C.VOTE_FACTOR = 1             #Vote factor [default: 1]')
C.CLUSTER_SAMPLING= 'vote_fps'#Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
C.AP_IOU_THRESH =0.25         #AP IoU threshold [default: 0.25]')
C.MAX_EPOCH =180              #Epoch to run [default: 180]')
C.BATCH_SIZE=8                #Batch Size during training [default: 8]')
C.LEARNING_RATE=0.001         #Initial learning rate [default: 0.001]')
C.RATIO = 1                   #Fractional [default: 1]')
C.WEIGHT_DECAY = 0            #Optimization L2 weight decay [default: 0]')
C.BN_DECAY_STEP = 20          #Period of BN decay (in epochs) [default: 20]')
C.BN_DECAY_RATE =0.5          #Decay rate for BN decay [default: 0.5]')
C.FIXED_LR=0                  #Fixed LR for not doing scheduling')
C.LR_DECAY_STEPS="80,120,160,200,240,280,320"#When to decay the learning rate (in epochs) [default: 80,120,160]')
C.LR_DECAY_RATES="0.1,0.1,0.1,0.1,0.1,0.1,0.1" #Decay rates for lr decay [default: 0.1,0.1,0.1]')
C.USE_COLOR=False             #Use RGB color in input
C.OVERWRITE = False           #Overwrite existing log and dump folders
C.START_ITER = 0              #Starting iterationg
C.DUMP_RESULTS = False        #Dump results.
C.NUM_VAL_BATCHES =-1         #num val batches
C.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

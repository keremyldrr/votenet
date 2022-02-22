import os
import sys
import numpy as np
from datetime import datetime
import importlib.util
import argparse



# pytorch stuff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
BASE_DIR = os.path.dirname(os.path.abspath("/home/yildirir/workspace/votenet/README.md"))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# project stuff

from dump_helper import dump_results_for_sanity_check
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
from  initialization_utils import initialize_dataloader,initialize_model, log_string

def save_datas(FLAGS):
    for  i in range(1):
        for T in FLAGS.TEST_DATALOADERS:

            for batch_idx, batch_data_label in enumerate(T):
                # for key in batch_data_label:
                #     batch_data_label[key] = batch_data_label[key].to(FLAGS.DEVICE)

                # Forward pass
                dump_results_for_sanity_check(batch_data_label,FLAGS.DUMP_DIR,FLAGS.DATASET_CONFIG,prefix ="thresh_{}_{}_".format(T.dataset.thresh,str(i)))

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("C",args.config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    FLAGS = mod.C

    initialize_dataloader(FLAGS)
    save_datas(FLAGS)

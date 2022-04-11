# Other python libraries
import numpy as np
import sys
import os
import importlib


sys.path.append("pointnet2")
## PyTorch Stuff

from torch import nn
from pytorch_utils import BNMomentumScheduler
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

sys.path.append("models")

##  Dataset classes
sys.path.append("sunrgbd")
sys.path.append("scannet")
from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset
from model_util_sunrgbd import SunrgbdDatasetConfig
from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
from model_util_scannet import ScannetDatasetConfig
from scannet_frames_dataset import ScannetDetectionFramesDataset, MAX_NUM_OBJ
from model_util_scannet import ScannetDatasetConfig
from tf_visualizer import Visualizer as TfVisualizer


def log_string(logger, out_str):
    logger.write(out_str + "\n")
    logger.flush()
    print(out_str)


def my_worker_init_fn(worker_id):
    # np.random.seed(np.random.get_state()[1][0] + worker_id)

    np.random.seed(1)  # np.random.get_state()[1][0] + worker_id)


def initialize_dataloader(FLAGS):
    if not os.path.exists(FLAGS.LOG_DIR):
        os.makedirs(FLAGS.LOG_DIR)
    FLAGS.LOGGER = open(os.path.join(FLAGS.LOG_DIR, "log_train.txt"), "a")
    FLAGS.LOGGER.write(str(FLAGS) + "\n")  # make this prettier and readable
    # Create Dataset and Dataloader
    if FLAGS.DATASET == "sunrgbd":
        DATASET_CONFIG = SunrgbdDatasetConfig()
        TRAIN_DATASET = SunrgbdDetectionVotesDataset(
            "train",
            num_points=FLAGS.NUM_POINTS,
            augment=True,
            use_color=FLAGS.USE_COLOR,
            use_height=(not FLAGS.NO_HEIGHT),
            use_v1=(not FLAGS.USE_SUNRGBD_V2),
        )
        TEST_DATASET = SunrgbdDetectionVotesDataset(
            "val",
            num_points=FLAGS.NUM_POINTS,
            augment=False,
            use_color=FLAGS.USE_COLOR,
            use_height=(not FLAGS.NO_HEIGHT),
            use_v1=(not FLAGS.USE_SUNRGBD_V2),
        )
    elif FLAGS.DATASET == "scannet":
        DATASET_CONFIG = ScannetDatasetConfig()
        if FLAGS.RATIO == 1:
            TRAIN_DATASET = ScannetDetectionDataset(
                "train",
                num_points=FLAGS.NUM_POINTS,
                augment=True,
                use_color=FLAGS.USE_COLOR,
                use_height=(not FLAGS.NO_HEIGHT),
                custom_path=FLAGS.CUSTOM_PATH,
            )
        else:
            TRAIN_DATASET = ScannetDetectionDataset(
                "fractional_train",
                num_points=FLAGS.NUM_POINTS,
                augment=True,
                use_color=FLAGS.USE_COLOR,
                use_height=(not FLAGS.NO_HEIGHT),
                ratio=FLAGS.RATIO,
                custom_path=FLAGS.CUSTOM_PATH,
            )

        TEST_DATASET = ScannetDetectionDataset(
            "val",
            num_points=FLAGS.NUM_POINTS,
            augment=False,
            use_color=FLAGS.USE_COLOR,
            use_height=(not FLAGS.NO_HEIGHT),
        )

    elif FLAGS.DATASET == "scannet_frames":
        # sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        # sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        DATASET_CONFIG = ScannetDatasetConfig()
        data_setting = {
            "dataset_path": "/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/",
            "train_source": "/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/train_frames.txt",
            "eval_source": "/home/yildirir/workspace/kerem/TorchSSC/DATA/ScanNet/val_frames.txt",
            "frames_path": "/home/yildirir/workspace/kerem/TorchSSC/DATA/scannet_frames_25k/",
        }
        TEST_DATASET = ScannetDetectionFramesDataset(
            data_setting,
            split_set="val",
            num_points=FLAGS.NUM_POINTS,
            use_color=False,
            use_height=True,
            augment=False,
            overfit=FLAGS.OVERFIT,
            center_noise_var=FLAGS.SIGMA,
        )
        TRAIN_DATASET = ScannetDetectionFramesDataset(
            data_setting,
            split_set="train",
            num_points=FLAGS.NUM_POINTS,
            use_color=False,
            use_height=True,
            augment=False,
            overfit=FLAGS.OVERFIT,
            center_noise_var=FLAGS.SIGMA,
        )
    else:
        print("Unknown dataset %s. Exiting..." % (FLAGS.DATASET))
        exit(-1)
    # log_string(str(len(TRAIN_DATASET)))
    FLAGS.TRAIN_DATALOADER = DataLoader(
        TRAIN_DATASET,
        batch_size=FLAGS.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        worker_init_fn=my_worker_init_fn,
    )
    FLAGS.TEST_DATALOADER = DataLoader(
        TEST_DATASET,
        batch_size=FLAGS.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        worker_init_fn=my_worker_init_fn,
    )
    FLAGS.DATASET_CONFIG = DATASET_CONFIG
    print(len(FLAGS.TRAIN_DATALOADER), len(FLAGS.TEST_DATALOADER))
    # Init the model and optimzier


def initialize_model(FLAGS):

    MODEL = importlib.import_module(FLAGS.MODEL)  # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_input_channel = int(FLAGS.USE_COLOR) * 3 + int(not FLAGS.NO_HEIGHT) * 1
    # TFBoard Visualizers
    FLAGS.TRAIN_VISUALIZER = TfVisualizer(FLAGS, "train")
    FLAGS.TEST_VISUALIZER = TfVisualizer(FLAGS, "test")
    print(FLAGS)
    if FLAGS.MODEL == "boxnet":
        Detector = MODEL.BoxNet
    elif FLAGS.MODEL == "votenet":
        Detector = MODEL.VoteNet
    else:
        Detector = MODEL.ChairNet
    FLAGS.MODEL = MODEL
    net = Detector(
        num_class=FLAGS.DATASET_CONFIG.num_class,
        num_heading_bin=FLAGS.DATASET_CONFIG.num_heading_bin,
        num_size_cluster=FLAGS.DATASET_CONFIG.num_size_cluster,
        mean_size_arr=FLAGS.DATASET_CONFIG.mean_size_arr,
        num_proposal=FLAGS.NUM_TARGET,
        input_feature_dim=num_input_channel,
        vote_factor=FLAGS.VOTE_FACTOR,
        sampling=FLAGS.CLUSTER_SAMPLING,
        log_var=FLAGS.LOG_VAR,
    )

    if torch.cuda.device_count() > 1:
        log_string(FLAGS.LOGGER, "Let's use %d GPUs!" % (torch.cuda.device_count()))
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net)
    net.to(device)

    criterion = MODEL.get_loss
    # Load the Adam optimizer
    optimizer = optim.Adam(
        net.parameters(), lr=FLAGS.LEARNING_RATE, weight_decay=FLAGS.WEIGHT_DECAY
    )
    if FLAGS.FIXED_LR != 0:
        BASE_LEARNING_RATE = FLAGS.FIXED_LR
    BN_DECAY_STEP = FLAGS.BN_DECAY_STEP
    BN_DECAY_RATE = FLAGS.BN_DECAY_RATE
    LR_DECAY_STEPS = [int(x) for x in FLAGS.LR_DECAY_STEPS.split(",")]
    LR_DECAY_RATES = [float(x) for x in FLAGS.LR_DECAY_RATES.split(",")]
    assert len(LR_DECAY_STEPS) == len(LR_DECAY_RATES)

    # Prepare LOG_DIR and DUMP_DIR
    if os.path.exists(FLAGS.LOG_DIR) and FLAGS.OVERWRITE:
        print(
            "Log folder %s already exists. Are you sure to overwrite? (Y/N)"
            % (FLAGS.LOG_DIR)
        )
        c = input()
        if c == "n" or c == "N":
            print("Exiting..")
            exit()
        elif c == "y" or c == "Y":
            print("Overwrite the files in the log and dump folers...")
            os.system("rm -r %s %s" % (FLAGS.LOG_DIR, FLAGS.DUMP_DIR))

    if not os.path.exists(FLAGS.LOG_DIR):
        os.mkdir(FLAGS.LOG_DIR)
    LOG_FOUT = open(os.path.join(FLAGS.LOG_DIR, "log_train.txt"), "a")
    LOG_FOUT.write(str(FLAGS) + "\n")
    # log_string(str(FLAGS)) ## TODO:   Properly print out configuration

    if FLAGS.DUMP_RESULTS:
        if not os.path.exists(FLAGS.DUMP_DIR):
            os.mkdir(FLAGS.DUMP_DIR)

    # Init datasets and dataloaders
    # Decay Batchnorm momentum from 0.5 to 0.999
    # note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum

    # Load checkpoint if there is any
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    start_epoch = 0
    if FLAGS.CHECKPOINT_PATH is not None and os.path.isfile(FLAGS.CHECKPOINT_PATH):
        checkpoint = torch.load(FLAGS.CHECKPOINT_PATH)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        log_string(
            FLAGS.LOGGER,
            "-> loaded checkpoint %s (epoch: %d)"
            % (FLAGS.CHECKPOINT_PATH, start_epoch),
        )

    FLAGS.START_EPOCH = start_epoch
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MAX = 0.001
    bn_lbmd = lambda it: max(
        BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX
    )
    bnm_scheduler = BNMomentumScheduler(
        net, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1
    )

    FLAGS.MODEL = MODEL

    return net, criterion, optimizer, bnm_scheduler

# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import time
import os
import sys
from time import thread_time
import numpy as np
import cv2
from torch.utils.data import Dataset
from sc_utils import get_axis_aligned_matrix, get_camera_pose, get_instance_and_semantic_pcd_from_boxes, get_camera_pose, get_instance_boxes, get_pcd_from_depth, corners_to_obb
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import pandas as pd
from sc_utils import get_axis_aligned_matrix, get_camera_pose, get_instance_and_semantic_pcd_from_boxes
from model_util_scannet import ScannetDatasetConfig

DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


class ScannetDetectionFramesDataset(Dataset):

    def __init__(self,
                 setting,
                 split_set='train',
                 num_points=20000,
                 use_color=False,
                 use_height=False,
                 augment=False,
                 thresh = 0.3):
        """[summary]

        Args:
            split_set (str, optional): [description]. Defaults to 'train'.
            num_points (int, optional): [description]. Defaults to 20000.
            use_color (bool, optional): [description]. Defaults to False.
            use_height (bool, optional): [description]. Defaults to False.
            augment (bool, optional): [description]. Defaults to False.
            rot ([type], optional): [description]. Defaults to None.
from pandas.core.dtypes.common import classes
            ratio ([type], optional): [description]. Defaults to None.
            custom_path ([type], optional): [description]. Defaults to None.
        """
        self._path = setting['dataset_path']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._frames_path = setting["frames_path"]
        self.thresh = thresh
        self.scan_names = self._get_file_names(split_set,thresh)
                 

        self.num_points = num_points
        self._file_length = len(self.scan_names)
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment

    def __len__(self):
        return len(self.scan_names)


    def _get_file_names(self, split_name,thresh=.3):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()
        #files = files[:500]
        print("Recreating files")  
        st = time.time()
        for item in files[:]:
            item = item.strip()
            item = item.split('\t')
            img_name = item[0]
            item_idx = img_name
            scene_name = item_idx[:item_idx.rfind("_")]
            unformatted = item_idx[item_idx.rfind("_") + 1:]

            instancedir = os.path.join(self._path, scene_name, "PCD",
                                   "instances_{}".format(unformatted))

            objs_in_scene = os.listdir(instancedir)        
            # instances = get_instance_boxes(instancedir,with_classes=False,thresh=thresh)
            
            scores = np.array([float(a[:-4].split("_")[5]) for a in objs_in_scene])
            if len(scores[scores > thresh]) > 0:
                file_names.append([img_name,None])
        with open("{}_{}.txt".format(str(thresh),split_name),"w") as out:
            for f in file_names:
                out.write(f[0] + "\n")
        print("Dataset created with size ", len(file_names), " took ",  time.time() - st, "seconds") 
        return file_names

    def __getitem__(self, idx):

        item_idx = self.scan_names[idx][0]
        scene_name = item_idx[:item_idx.rfind("_")]
        formatted = format(100 * int(item_idx[item_idx.rfind("_") + 1:]),
                           "06d")
        unformatted = item_idx[item_idx.rfind("_") + 1:]
        axis_align_matrix = get_axis_aligned_matrix(scene_name)
        # pcd_path = os.path.join(self._path,scene_name,"PCD","frame{}.ply".format(formatted)) # maybe reading with trimesh would be faster

        instancedir = os.path.join(self._path, scene_name, "PCD",
                                   "instances_{}".format(unformatted))

        poses = sorted(
            os.listdir(os.path.join(self._frames_path, scene_name, "pose/")))

        depths = sorted(
            os.listdir(os.path.join(self._frames_path, scene_name, "depth/")))
        pose_path = os.path.join(self._frames_path, scene_name,
                                 "pose/" + poses[int(unformatted)])
        depth_image_path = os.path.join(self._frames_path, scene_name,
                                        "depth/" + depths[int(unformatted)])
        camera_pose = get_camera_pose(pose_path)
        path_to_depth_intr = os.path.join(self._frames_path, scene_name,
                                          "intrinsics_depth.txt")
        depth_intrinsic = pd.read_csv(path_to_depth_intr,
                                      header=None,
                                      delimiter=" ").values[:, :-1]

        depth_image = np.array(cv2.imread(depth_image_path, -1),
                               dtype=np.float32) / 1000
        pts3d, valid_depth_inds = get_pcd_from_depth(
            depth_image=depth_image, depth_intrinsic=depth_intrinsic)
        transform = np.matmul(axis_align_matrix, camera_pose)
        rot = transform[:3, :3]
        tl = transform[:-1, 3]
        pts = np.matmul(rot, pts3d[valid_depth_inds].T).T + tl
        # worldToCam = get_grid_to_camera(camera_pose=camera_pose,axis_align_matrix=axis_align_matrix)
        boxes, classes = get_instance_boxes(instancedir=instancedir,
                                            with_classes=True)

        instance_labels, semantic_labels = get_instance_and_semantic_pcd_from_boxes(points=pts, boxes=boxes, classes=classes)
        VOX_SIZE = 1  # double check
        grid_shape = np.array([60, 36, 60])
        pts = (pts[:, [0, 1, 2]] / VOX_SIZE)

        ptrans = grid_shape / 2 - pts.mean(0)
        pts = ((pts + ptrans))  #.astype(int) #Not yet integet

        boxes_in_grid = []
        for b in boxes:

            bpts = (b[:, [0, 1, 2]]/VOX_SIZE)

            bpts = (bpts + ptrans)
            boxes_in_grid.append(bpts)


        instance_bboxes = corners_to_obb(boxes_in_grid, classes=classes
        )  # Convert boxes from 8 corners to Nx7 representation
        ## TODO: read the pose and instances

        # label the input points with selected boxes. Kutularin icindeki noktalar kutunun classina ait olacak
        # boxlara ait classlar semantic label. her boxun unique instance indexi olacak
        # boxes should be in same coordsinate system as the scene
        if not self.use_color:
            point_cloud = pts[:, 0:3]  # do not use color for now
            pcl_color = pts[:, 3:6]
        else:
            point_cloud = torch.Tensor(pts[:, 0:6])  # TODO: add color
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1)], 1)

            # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ, ))
        angle_residuals = np.zeros((MAX_NUM_OBJ, ))
        size_classes = np.zeros((MAX_NUM_OBJ, ))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        point_cloud, choices = pc_util.random_sampling(point_cloud,
                                                       self.num_points,
                                                       return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]

        pcl_color = pcl_color[choices]

        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label.
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        for i_instance in np.unique(instance_labels):
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label
            if semantic_labels[ind[0]] in DC.class_ids:
                x = point_cloud[ind, :3]
                center = 0.5 * (x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical
        class_ind = [
            np.where(DC.class_ids == x)[0][0] for x in instance_bboxes[:, -1]
        ]

        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind,:]
        ret_dict = {}
        point_cloud = torch.from_numpy(point_cloud)
        ret_dict['point_clouds'] = point_cloud.float() 
        ret_dict['center_label'] = torch.from_numpy(target_bboxes.astype(np.float32)[:, 0:3])
        ret_dict['heading_class_label'] =torch.from_numpy(angle_classes.astype(np.int64))

        ret_dict['heading_residual_label'] = torch.from_numpy(angle_residuals.astype(np.float32))

        ret_dict['size_class_label'] = torch.from_numpy(size_classes.astype(np.int64))

        ret_dict['size_residual_label'] = torch.from_numpy(size_residuals.astype(np.float32))


        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        #TODO: make sure classes are correct 
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
            [x for x in instance_bboxes[:,-1][0:instance_bboxes.shape[0]]]
        ret_dict['sem_cls_label'] = torch.from_numpy(target_bboxes_semcls.astype(np.int64))

        ret_dict['box_label_mask'] =torch.from_numpy(target_bboxes_mask.astype(np.float32))

        ret_dict['vote_label'] =torch.from_numpy(point_votes.astype(np.float32))

        ret_dict['vote_label_mask'] = torch.from_numpy(point_votes_mask.astype(np.int64))

        ret_dict['scan_idx'] = torch.from_numpy(np.array(idx).astype(np.int64))

        ret_dict['pcl_color'] = (pcl_color)
        ret_dict['name'] = item_idx
        return ret_dict


############# Visualizaion ########


def viz_votes(pc, point_votes, point_votes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask == 1)
    pc_obj = pc[inds, 0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds, 0:3]
    pc_util.write_ply(pc_obj, 'pc_obj{}.ply'.format(name))
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1{}.ply'.format(name))


def viz_obb(pc,
            label,
            mask,
            angle_classes,
            angle_residuals,
            size_classes,
            size_residuals,
            name=''):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i, 0:3]
        heading_angle = 0  # hard code to 0
        box_size = DC.mean_size_arr[size_classes[i], :] + size_residuals[i, :]
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs{}.ply'.format(name))
    pc_util.write_ply(label[mask == 1, :], 'gt_centroids{}.ply'.format(name))


if __name__ == '__main__':
    dset = ScannetDetectionDataset(use_height=True, num_points=40000)
    for i_example in range(4):
        example = dset.__getitem__(1)
        pc_util.write_ply(example['point_clouds'],
                          'pc_{}.ply'.format(i_example))
        viz_votes(example['point_clouds'],
                  example['vote_label'],
                  example['vote_label_mask'],
                  name=i_example)
        viz_obb(pc=example['point_clouds'],
                label=example['center_label'],
                mask=example['box_label_mask'],
                angle_classes=None,
                angle_residuals=None,
                size_classes=example['size_class_label'],
                size_residuals=example['size_residual_label'],
                name=i_example)

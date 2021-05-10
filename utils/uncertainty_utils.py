import numpy as np
import torch
from models.ap_helper import softmax
from box_util import box3d_iou
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
#TODO: make dimensions nicer
def apply_softmax(samples):
    for e in samples:
        e["sm_objectness_scores"] = (softmax(e["objectness_scores"].cpu().squeeze(0).detach().numpy()))
        e["sm_sem_cls_scores"] = (softmax(e["sem_cls_scores"].cpu().squeeze(0).detach().numpy()))

def semantic_cls_uncertainty(samples):
    mc_cls = np.array([e["sm_sem_cls_scores"] for e in samples])
    expected_p = np.mean(mc_cls, axis=0)
    predictive_entropy = -np.sum(expected_p *np.log(expected_p), axis=-1)
    MC_entropy = np.sum(mc_cls * np.log(mc_cls),axis=-1)
    expected_entropy = -np.mean(MC_entropy, axis=0)
    mi = predictive_entropy - expected_entropy
    normalized_mi = map_zero_one(mi)
    mi_mask = np.logical_not(normalized_mi > normalized_mi.mean())
    return mi_mask,normalized_mi


def objectness_uncertainty(samples):
    mc_objs = np.array([e["sm_objectness_scores"] for e in samples])
    expected_p_obj = np.mean(mc_objs, axis=0)
    predictive_entropy_obj = -np.sum(expected_p_obj *np.log(expected_p_obj), axis=-1)
    MC_entropy_obj = np.sum(mc_objs * np.log(mc_objs),axis=-1)
    expected_entropy_obj = -np.mean(MC_entropy_obj, axis=0)
    mi_obj = predictive_entropy_obj - expected_entropy_obj
    normalized_mi_obj = map_zero_one(mi_obj)
    mi_obj_mask = np.logical_not(normalized_mi_obj > normalized_mi_obj.mean())
    return mi_obj_mask,normalized_mi_obj
    


def center_uncertainty(samples):
    expected_centers = torch.zeros([256,3]).unsqueeze(0)
    centers = []
    for e in samples:
        centers.append(e["center"])
    center_variance = np.var(np.array([c.squeeze(0).cpu().detach().numpy() for c in centers]),axis=0)
    center_mean =np.mean(np.array([c.squeeze(0).cpu().detach().numpy() for c in centers]),axis=0)
    normalized_center_variance = map_zero_one(center_variance)
    center_variance_mask = normalized_center_variance > normalized_center_variance.mean()
    return center_variance_mask,normalized_center_variance
    #entropy
def box_size_uncertainty(samples):
    expected_centers = torch.zeros([256]).unsqueeze(0)
    boxes = []
    for e in samples:
        boxes.append(e["pred_box_sizes"])
    box_variance = np.var(np.array([c.squeeze(0).cpu().detach().numpy() for c in boxes]),axis=0)
    box_mean =np.mean(np.array([c.squeeze(0).cpu().detach().numpy() for c in boxes]),axis=0)
    normalized_box_variance = map_zero_one(box_variance)
    box_variance_mask = np.logical_not(normalized_box_variance > normalized_box_variance.mean())
    return box_variance_mask,normalized_box_variance
    


def compute_objectness_accuracy(samples):
    """
    Goes over the raw predicted boxes and for every prediction computes iou with all gt boxes. 
    If it's above a threshold (0.5 for now), then it is counted as a correct match.(TP) Accuracy will be computed as
    TP/256.
    """
    for end_points in samples:
        pred_boxes = end_points["raw_pred_boxes"]
        pred_box_sizes = end_points["pred_box_sizes"]
        gt_boxes = end_points["raw_gt_boxes"]
        gt_box_sizes = end_points["gt_box_sizes"]
        batchSize = len(pred_boxes)
        iou_map = np.zeros([batchSize,256,64])
        size_map = np.zeros([batchSize,256,64])
        accs = []
        temp = 0
        for i in range(batchSize):
            temp = 0
            for idx,b in enumerate(pred_boxes[i]):
                for idy,gt in enumerate(gt_boxes[i]):
                    iou,iou_2d = box3d_iou(b, gt)
                    iou_map[i,idx,idy] = iou * (np.int(iou >= 0.25))
                    size_map[i,idx,idy] = np.abs(pred_box_sizes[i][idx] - gt_box_sizes[i][idy])
                    if iou >= 0.25:
                        temp+=1
                    else:
                        size_map[i,idx,idy] = np.inf
                     
                    
            iou_mask = np.argmax(iou_map[i],axis = 1)
           
           #TODO: This is very wrong
            accs.append(temp/256)


    return np.array(accs).mean()    

def compute_iou_masks(samples):
    """
    Compares all predicted boxes with ground truth boxes and marks the ones which have enough overlap with any gt box
    """
    iou_masks = []
    for end_points in samples:
        pred_boxes = end_points["raw_pred_boxes"]
        
        gt_boxes = end_points["raw_gt_boxes"]
        
        batchSize = len(pred_boxes)
        iou_map = np.zeros([batchSize,256,64])
        
        masks = []
        temp = 0
        for i in range(batchSize):
            temp = 0
            for idx,b in enumerate(pred_boxes[i]):
                for idy,gt in enumerate(gt_boxes[i]):
                    iou,iou_2d = box3d_iou(b, gt)
                    iou_map[i,idx,idy] = iou * (np.int(iou >= 0.25)) #accept as match if iou > 0.25
                    
                    
                        
                    
            iou_mask = np.array(np.sum(iou_map[i],axis = 1) > 0,dtype=np.int)
            
        end_points["iou_mask"] = iou_mask
        iou_masks.append(iou_mask)
    return iou_masks


def map_zero_one(A):
    """
    Maps a given array to the interval [0,1]
    """
    A_std = (A - A.min())/(A.max()-A.min())
    return A_std * (A.max() - A.min()) + A.min()

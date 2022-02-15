import numpy as np
import trimesh
from numpy.core.defchararray import center
import torch
import os
import sys

from box_util import box3d_iou
from sklearn.preprocessing import MinMaxScaler

# THRESHOLD = 0.5
THRESHOLD = lambda x: x.mean() + 2*x.var()

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs

#TODO: make dimensions nicer
def apply_softmax(samples):
    for e in samples:
        e["sm_objectness_scores"] = (softmax(e["objectness_scores"].cpu().squeeze(0).detach().numpy()))
        e["sm_sem_cls_scores"] = (softmax(e["sem_cls_scores"].cpu().squeeze(0).detach().numpy()))

def accumulate_mc_samples(mc_samples,classification=None):
    """
    This function accumulates everything in the end_points for the evaluation

    """
    all_centers = [e["center"] for e in mc_samples]
    all_size_scores = [e["size_scores"] for e in mc_samples]
    all_size_residuals = [e["size_residuals"] for e in mc_samples]
    all_cls_scores = [e["sem_cls_scores"] for e in mc_samples]
    all_head_residuals = [e["heading_residuals"] for e in mc_samples]
    all_head_scores = [e["heading_scores"] for e in mc_samples]
    all_objectness_scores = [e["objectness_scores"] for e in mc_samples]

    all_point_clouds = [e["point_clouds"] for e in mc_samples]
    mean_centers=torch.mean(torch.stack(all_centers),dim = 0)
    mean_size_scores =torch.mean(torch.stack(all_size_scores),dim = 0)
    mean_cls_scores =torch.mean(torch.stack(all_cls_scores),dim = 0)
    mean_heading_residuals =torch.mean(torch.stack(all_head_residuals),dim = 0)
    mean_size_residuals =torch.mean(torch.stack(all_size_residuals),dim = 0)
    mean_heading_scores =torch.mean(torch.stack(all_head_scores),dim = 0)
    mean_objectness_scores =torch.mean(torch.stack(all_objectness_scores),dim = 0)

    mean_point_clouds =torch.mean(torch.stack(all_point_clouds),dim = 0)
    apply_softmax(mc_samples)

    mean_end_points = {}
    mean_end_points["center"] = mean_centers
    
    mean_end_points["size_scores"] =mean_size_scores
    mean_end_points["objectness_scores"] =mean_objectness_scores
    mean_end_points["size_residuals"] =mean_size_residuals
    mean_end_points["sem_cls_scores"] = mean_cls_scores
    mean_end_points["point_clouds"] = mean_point_clouds
    
    mean_end_points["heading_residuals"] =mean_heading_residuals
    mean_end_points["heading_scores"] = mean_heading_scores
    if classification is None:
        _,mean_end_points["semantic_cls_entropy"] = semantic_cls_uncertainty(mc_samples)
        _,mean_end_points["objectness_entropy"] = objectness_uncertainty(mc_samples)
    else:
        _,mean_end_points["semantic_cls_entropy"] = semantic_cls_uncertainty(mc_samples,classification=classification)
        _,mean_end_points["objectness_entropy"] = objectness_uncertainty(mc_samples,classification=None)
    # mean_end_points["center_variance"] = center_uncertainty(mc_samples)
    return mean_end_points
def accumulate_scores(mc_samples):
    """
    This function accumulates everything in the end_points for the evaluation

    """
    
    
    all_cls_scores = [e["sem_cls_scores"] for e in mc_samples]
    
    all_objectness_scores = [e["objectness_scores"] for e in mc_samples]

    all_point_clouds = [e["point_clouds"] for e in mc_samples]
    
    mean_cls_scores =torch.mean(torch.stack(all_cls_scores),dim = 0)
    
    mean_objectness_scores =torch.mean(torch.stack(all_objectness_scores),dim = 0)

    mean_point_clouds =torch.mean(torch.stack(all_point_clouds),dim = 0)
    apply_softmax(mc_samples)

    mean_end_points = {}
    
    
    mean_end_points["objectness_scores"] =mean_objectness_scores
    mean_end_points["sem_cls_scores"] = mean_cls_scores
    mean_end_points["point_clouds"] = mean_point_clouds
    _,mean_end_points["semantic_cls_entropy"] = semantic_cls_uncertainty(mc_samples)
    _,mean_end_points["objectness_entropy"] = objectness_uncertainty(mc_samples)
    # mean_end_points["center_variance"] = center_uncertainty(mc_samples)
    return mean_end_points

def semantic_cls_uncertainty(samples,threshold = None,classification=None):
    mc_cls = np.array([e["sm_sem_cls_scores"] for e in samples])
    expected_p = np.mean(mc_cls, axis=0)
    predictive_entropy = -np.sum(expected_p *np.log(expected_p), axis=-1)
    MC_entropy = np.sum(mc_cls * np.log(mc_cls),axis=-1)
    expected_entropy = -np.mean(MC_entropy, axis=0)
    if classification is None: 
        mi = predictive_entropy - expected_entropy
    else:
        mi = predictive_entropy
    normalized_mi = np.zeros_like(mi)
    for i in range(len(mi)):
        # print("item ",(samples[0]["names"][i]))
        normalized_mi[i] = map_zero_one(mi[i])
        # trimesh.points.PointCloud(samples[0]["point_clouds"][i,:,:3].cpu().numpy()).export("{}.ply".format(samples[0]["names"][i]))
        
    # # print("Mean classification entropy", normalized_mi.mean())
    # print(normalized_mi)
    # raise NotImplementedError
    if threshold != None:
        mi_mask =np.array((normalized_mi < threshold),dtype=np.int)
    else:
        mi_mask = np.logical_not(normalized_mi > THRESHOLD(normalized_mi))
    return mi_mask,normalized_mi


def objectness_uncertainty(samples,threshold = None,classification=None):
    mc_objs = np.array([e["sm_objectness_scores"] for e in samples])
    expected_p_obj = np.mean(mc_objs, axis=0)
    predictive_entropy_obj = -np.sum(expected_p_obj *np.log(expected_p_obj), axis=-1)
    MC_entropy_obj = np.sum(mc_objs * np.log(mc_objs),axis=-1)
    expected_entropy_obj = -np.mean(MC_entropy_obj, axis=0)
    if classification is None:
        mi_obj = predictive_entropy_obj - expected_entropy_obj
    else:
        mi_obj = predictive_entropy_obj
    normalized_mi_obj = np.zeros_like(mi_obj) 
    for i in range(len(mi_obj)):
        # print("item ",(samples[0]["names"][i]))
        normalized_mi_obj[i] = map_zero_one(mi_obj[i])
    # print("Mean objectness entropy",normalized_mi_obj.mean())

    if threshold != None:
        mi_obj_mask = np.array((normalized_mi_obj < threshold),dtype=np.int)
    else:
        mi_obj_mask = (normalized_mi_obj > THRESHOLD(normalized_mi_obj))
    return mi_obj_mask,normalized_mi_obj
    


def center_uncertainty(samples):
    centers = []
    logvars = []
    logvars2 = []
    num_batch = samples[0]["center"].shape[0]
    center_variances = np.zeros([num_batch,256])#.unsqueeze(0)
    
    center_variances2 = np.zeros([num_batch,256])#.unsqueeze(0)
    for i in range(num_batch):
        centers.clear()
        for s in samples:
            centers.append(s["center"][i,:])
            logvars.append(np.exp(s["log_var_center"][i,:].detach().cpu().numpy())*np.exp(s["log_var_center"][i,:].detach().cpu().numpy()))
            logvars2.append(s["log_var_center"][i,:].detach().cpu().numpy()*s["log_var_center"][i,:].detach().cpu().numpy())

        center_mean =np.mean(np.array([(c*c).squeeze(0).cpu().detach().numpy() for c in centers]),axis=0)
        center_sq_sum =np.mean(np.array([c.squeeze(0).cpu().detach().numpy() for c in centers]),axis=0)
        center_sq_sum *= center_sq_sum  
        # center_variances[i] = map_zero_one(np.sum(center_mean,axis=1) - np.sum(center_sq_sum,axis=1) +  np.sum(np.array(logvars),0))
        # center_variances[i] = torch.sigmoid(torch.from_numpy(np.sum(center_mean,axis=1) - np.sum(center_sq_sum,axis=1) +  np.sum(np.array(logvars),0))).numpy()
        center_variances[i] =torch.from_numpy(np.sum(center_mean,axis=1) - np.sum(center_sq_sum,axis=1) +  np.sum(np.array(logvars),0)).numpy()
        
        center_variances2[i] = np.var(np.array([c for c in logvars2]),axis=0)
        # center_mean =np.mean(np.array([c.squeeze(0).cpu().detach().numpy() for c in centers]),axis=0)
        # normalized_center_variance = map_zero_one(center_variance)
        # center_variance_mask = normalized_center_variance > normalized_center_variance.mean()
    
    
    return center_variances
    #entropy
def box_size_uncertainty(samples,threshold = None):
    expected_centers = torch.zeros([256]).unsqueeze(0)
    boxes = []
    for e in samples:
        boxes.append(e["pred_box_sizes"])
    box_variance = np.var(np.array([c.squeeze(0).cpu().detach().numpy() for c in boxes]),axis=0)
    box_mean =np.mean(np.array([c.squeeze(0).cpu().detach().numpy() for c in boxes]),axis=0)
    normalized_box_variance = map_zero_one(box_variance)
    print("Mean box size entropy",normalized_box_variance.mean())

    if threshold != None:
        box_variance_mask = np.array(normalized_box_variance < threshold,dtype=np.int)
    else:
        box_variance_mask = np.logical_not(normalized_box_variance > THRESHOLD(normalized_box_variance))
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

def compute_total_accuracy_from_masks(gt_mask,pred_mask):
    pass
def compute_per_class_accuracy_from_masks(gt_mask,pred_mask,cls_dict ):
    pass
def compute_iou_masks_with_classification(samples,class_acc):
    """
    Compares all predicted boxes with ground truth boxes and marks the ones which have enough overlap with any gt box
    """
    iou_masks = []
    cls_iou_masks = []

    for end_points in samples:
        pred_boxes = end_points["raw_pred_boxes"]
        sem_cls_probs = softmax(end_points['sem_cls_scores'].detach().cpu().numpy())  # B,num_proposal,10
        pred_sem_cls_prob = np.max(sem_cls_probs, -1)
        gt_boxes = end_points["raw_gt_boxes"]
        batchSize = len(pred_boxes)
        sem_cls_label = end_points['sem_cls_label']
        true_labels = np.ones([batchSize,256]) - 2 # array full of -1s non labeled
        iou_map = np.zeros([batchSize,256,64])
        cls_iou_map = np.zeros([batchSize,256,64])

        masks = []
        temp = 0
        for i in range(batchSize):
            temp = 0
            pred_labels = np.argmax(sem_cls_probs[i],axis=1)
            gt_labels = sem_cls_label[i]
            for idx,b in enumerate(pred_boxes[i]):
                for idy,gt in enumerate(gt_boxes[i]):
                    gt_label = gt_labels[idy].cpu().item()
                    pred_label = pred_labels[idx]
                    iou,iou_2d = box3d_iou(b, gt)
                    check = np.int(pred_label == gt_label)
                    iou_map[i,idx,idy] = iou * (np.int(iou >= 0.25)) #accept as match if iou > 0.25
                    cls_iou_map[i,idx,idy] = iou * (np.int(iou >= 0.25))*check #accept as match if iou > 0.25
                    if( iou >= 0.25):
                        if check == 1:
                            true_labels[i,idx] = gt_label
                            class_acc[gt_label][0]+=1
                        else:
                            class_acc[gt_label][1]+=1
                        
            
            # unique, counts = np.unique(np.argmax(iou_map[i],1), return_counts=True)
            # gt_uniq,gt_counts = np.unique(gt_labels.cpu(),return_counts=True)
            # gt_result = np.column_stack((gt_uniq, gt_counts)) 
            # print ("GT",gt_result)
            # result = np.column_stack((unique, counts)) 
            # print ("Predicted",result)


            iou_mask = np.array(np.sum(iou_map[i],axis = 1) > 0,dtype=np.int)
            cls_iou_mask = np.array(np.sum(cls_iou_map[i],axis = 1) > 0,dtype=np.int)
            
        end_points["iou_mask"] = iou_mask
        end_points["cls_iou_mask"] = cls_iou_mask
        end_points["true_labels"] = true_labels

        iou_masks.append(iou_mask)
        cls_iou_masks.append(cls_iou_mask)
    iou_masks = [np.array(np.sum(np.array(iou_masks),axis=0) > len(iou_masks)/2,dtype=np.int)]
    cls_iou_masks = [np.array(np.sum(np.array(cls_iou_masks),axis=0) > len(cls_iou_masks)/2,dtype=np.int)]
    return cls_iou_masks,iou_masks

def map_zero_one(A):
    """
    Maps a given array to the interval [0,1]
    """
    # return A
    # print("MI max and min ",A.max(),A.min())
    A_std = (A - A.min())/(A.max()-A.min())
    retval = A_std * (A.max() - A.min()) + A.min()
    return A_std
    # return softmax(A)
    # return torch.sigmoid(torch.Tensor(A)).numpy()

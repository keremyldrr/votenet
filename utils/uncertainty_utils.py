import numpy as np
import torch
from models.ap_helper import softmax
from box_util import box3d_iou
from sklearn.preprocessing import MinMaxScaler

# THRESHOLD = 0.5
THRESHOLD = lambda x: x.mean() + 2*x.var()

scaler = MinMaxScaler()
#TODO: make dimensions nicer
def apply_softmax(samples):
    for e in samples:
        e["sm_objectness_scores"] = (softmax(e["objectness_scores"].cpu().squeeze(0).detach().numpy()))
        e["sm_sem_cls_scores"] = (softmax(e["sem_cls_scores"].cpu().squeeze(0).detach().numpy()))

def semantic_cls_uncertainty(samples,threshold = None):
    mc_cls = np.array([e["sm_sem_cls_scores"] for e in samples])
    expected_p = np.mean(mc_cls, axis=0)
    predictive_entropy = -np.sum(expected_p *np.log(expected_p), axis=-1)
    MC_entropy = np.sum(mc_cls * np.log(mc_cls),axis=-1)
    expected_entropy = -np.mean(MC_entropy, axis=0)
    mi = predictive_entropy - expected_entropy
    normalized_mi = map_zero_one(mi)
    print("Mean classification entropy", normalized_mi.mean())

    if threshold != None:
        mi_mask =np.array((normalized_mi < threshold),dtype=np.int)
    else:
        mi_mask = np.logical_not(normalized_mi > THRESHOLD(normalized_mi))
    return mi_mask,normalized_mi


def objectness_uncertainty(samples,threshold = None):
    mc_objs = np.array([e["sm_objectness_scores"] for e in samples])
    expected_p_obj = np.mean(mc_objs, axis=0)
    predictive_entropy_obj = -np.sum(expected_p_obj *np.log(expected_p_obj), axis=-1)
    MC_entropy_obj = np.sum(mc_objs * np.log(mc_objs),axis=-1)
    expected_entropy_obj = -np.mean(MC_entropy_obj, axis=0)
    mi_obj = predictive_entropy_obj - expected_entropy_obj
    normalized_mi_obj = map_zero_one(mi_obj)
    print("Mean objectness entropy",normalized_mi_obj.mean())

    if threshold != None:
        mi_obj_mask = np.array((normalized_mi_obj < threshold),dtype=np.int)
    else:
        mi_obj_mask = (normalized_mi_obj > THRESHOLD(normalized_mi_obj))
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
    A_std = (A - A.min())/(A.max()-A.min())
    retval = A_std * (A.max() - A.min()) + A.min()
    return retval 
    #return softmax(A)
    # return torch.sigmoid(torch.Tensor(A)).numpy()
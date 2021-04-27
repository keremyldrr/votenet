impoer numpy as np
import torch


def apply_softmax(samples):
    for e in samples:
        e["objectness_scores"] = softmax(e["objectness_scores"].cpu().detach().numpy())
        e["sem_cls_scores"] = softmax(e["sem_cls_scores"].cpu().detach().numpy())[0]



def semantic_cls_uncertainty(samples):
    mc_cls = np.array([e["sem_cls_scores"] for e in samples])
    expected_p = np.mean(mc_cls, axis=0)
    predictive_entropy = -np.sum(expected_p *np.log(expected_p), axis=-1)
    MC_entropy = np.sum(mc_cls * np.log(mc_cls),axis=-1)
    expected_entropy = -np.mean(MC_entropy, axis=0)
    mi = predictive_entropy - expected_entropy
    return mi.sum()


def objectness_uncertainty(samples):
    mc_objs = np.array([e["objectness_scores"] for e in samples])
    expected_p_obj = np.mean(mc_objs, axis=0)
    predictive_entropy_obj = -np.sum(expected_p_obj *np.log(expected_p_obj), axis=-1)
    MC_entropy_obj = np.sum(mc_objs * np.log(mc_objs),axis=-1)
    expected_entropy_obj = -np.mean(MC_entropy_obj, axis=0)
    mi_obj = predictive_entropy_obj - expected_entropy_obj
    return mi_obj.sum()




def center_uncertainty(samples):
    expected_centers = torch.zeros([256,3]).unsqueeze(0)
    centers = []
    for e in samples:
        centers.append(e["center"])
    center_variance = np.var(np.array([c.squeeze(0).cpu().detach().numpy() for c in centers]),axis=0)
    center_mean =np.mean(np.array([c.squeeze(0).cpu().detach().numpy() for c in centers]),axis=0)
    return center_variance.sum()

#entropy
def box_size_uncertainty(samples):
    expected_centers = torch.zeros([256]).unsqueeze(0)
    boxes = []
    for e in samples:
        boxes.append(e["pred_box_sizes"])
    box_variance = np.var(np.array([c.squeeze(0).cpu().detach().numpy() for c in boxes]),axis=0)
    box_mean =np.mean(np.array([c.squeeze(0).cpu().detach().numpy() for c in boxes]),axis=0)
    return center_variance.sum()
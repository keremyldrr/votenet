import numpy as np


class UncertaintyFilter:
    def __init__(self,name):
        self.name = name
        self.classification_class_dict = {}
        self.objectness_class_dict = {}
        self.fvector = None
        self.obj_accs = []
        self.cls_accs = []
        for i in range(19):
            self.classification_class_dict[i] = [0,0]
            self.objectness_class_dict[i] = [0,0]
        self.num_rejected = []
    #added another class for non existent objects
    def accumulate_scores_cls(self,end_points,cls_iou_mask):
        sem_cls_probs =end_points['sm_sem_cls_scores']  # B,num_proposal,10
        pred_labels = np.argmax(sem_cls_probs,axis=1)
        for i in range(len(self.fvector)):
            if self.fvector[i] == 1 and cls_iou_mask[i] == 1:
                pred_class = pred_labels[i]
                self.classification_class_dict[pred_class][0] =self.classification_class_dict[pred_class][0]+ 1
            elif self.fvector[i] == 0 and cls_iou_mask[i] == 1:
                label = end_points["true_labels"][0,i] #missed guess
                self.classification_class_dict[label][1] =self.classification_class_dict[label][1]  + 1
            elif self.fvector[i] == 1 and cls_iou_mask[i] == 0:
                self.classification_class_dict[pred_labels[i]][1] =self.classification_class_dict[pred_labels[i]][1]  +  1
                self.classification_class_dict[18][1] =self.classification_class_dict[18][1]  +  1
            else:
                self.classification_class_dict[18][0] =self.classification_class_dict[18][0]  +  1
        done = True

    # def accumulate_scores_obj(self,end_points,iou_mask):
    #     sem_cls_probs =end_points['sm_sem_cls_scores']  # B,num_proposal,10
    #     pred_labels = np.argmax(sem_cls_probs,axis=1)
    #     for i in range(len(self.fvector)):
    #         if self.fvector[i] == 1 and iou_mask[i] == 1:
    #             pred_class = pred_labels[i]
    #             self.classification_class_dict[pred_class][0] =self.classification_class_dict[pred_class][0]+ 1
    #         elif self.fvector[i] == 0 and iou_mask[i] == 1:
    #             label = end_points["true_labels"][0,i] #missed guess
    #             self.classification_class_dict[label][1] =self.classification_class_dict[label][1]  + 1
    #         elif self.fvector[i] == 1 and iou_mask[i] == 0:
    #             self.classification_class_dict[18][1] =self.classification_class_dict[18][1]  +  1
    #         else:
    #             self.classification_class_dict[18][0] =self.classification_class_dict[18][0]  +  1
    #     done = True

    def update(self,end_points,iou_masks,cls_iou_masks):
        self.num_rejected.append(np.count_nonzero(self.fvector == 0))
        self.obj_accs.append((len(iou_masks[0]) - np.sum(np.logical_xor(self.fvector, iou_masks[0])))/len(iou_masks[0]))
        self.cls_accs.append((len(cls_iou_masks[0]) - np.sum(np.logical_xor(self.fvector, cls_iou_masks[0])))/len(cls_iou_masks[0]))
        #Compute scores for each box here 
        #TODO: Accumulate classes as well when mergin the ,masks
        # self.accumulate_scores_cls(end_points,cls_iou_masks[0])

    def set_mask(self,fvector):
        self.fvector = fvector

    def log(self):
        print(self.name, "O: ",self.obj_accs[-1],"C: ",self.cls_accs[-1])
    def get_last_accs(self):
        return self.obj_accs[-1],self.cls_accs[-1]
    def dump_to_frame(self):
        pass
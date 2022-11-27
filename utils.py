import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random





def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def accuracy(predicted_label,ground_label):
    total_num = len(list(ground_label))
    acc = predicted_label == ground_label
    return acc.sum().float() / total_num * 100.0

def balance_accuracy(predicted_label,ground_label,num_cls=65):
    acc = []
    for cls in range(num_cls):
        part_pred_label = predicted_label[ground_label == cls]        
        part_ground_label = ground_label[ground_label == cls]
        part_acc =  accuracy(part_pred_label,part_ground_label) 
        acc.append(part_acc)
    
    return sum(acc) / num_cls

def class_accuracy(predicted_label,ground_label,num_cls=65):
    acc = []
    for cls in range(num_cls):
        part_pred_label = predicted_label[ground_label == cls]        
        part_ground_label = ground_label[ground_label == cls]
        part_acc =  accuracy(part_pred_label,part_ground_label) 
        acc.append(part_acc)
    
    return acc

class EntLoss(nn.Module):
    def __init__(self):
        super(EntLoss, self).__init__()

    def forward(self, pred):
        pred = F.softmax(pred, dim=1)
        b = - torch.mean(torch.sum(pred * (torch.log(pred + 1e-5)), 1))
        return b

    
class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type 

    def get_dist(self, pointA, pointB, cross=False):
        return getattr(self, self.dist_type)(
		pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 1.0 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            # NA = pointA.size(0)
            # NB = pointB.size(0)
            assert(pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))


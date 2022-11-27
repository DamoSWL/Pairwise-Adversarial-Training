

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from collections import Counter
import torch.nn.functional as F
from torch.optim import SGD


class PAT(object):
    def __init__(self,cfg=None):

        self.device = torch.device(cfg['gpu'] if torch.cuda.is_available() else 'cpu')
        self.target_label = None
        self.target_prob = None

        self.source_dataset = None
        self.target_dataset = None

        self.model = None
        self.class_num = cfg['class_num']

        self.thresh_prob_class = {}
        self.thresh_prob_pesudo = cfg['threshold_prob']
        self.max_loop = cfg['max_loop']
        self.pat_cof = cfg['pat_weight']
        self.rng = np.random.default_rng(42)
        
        self.num_per_cls = []

        self.epsilon = 1e-7


    def init_dataset(self,source_dataset,target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
       
        self.get_thresh_prob()
        

    def init_target_pseudo(self,num):
        self.target_label = torch.ones(num).long().fill_(-1)
        self.target_label = self.target_label.to(self.device)

        self.target_prob = torch.zeros(num).float().fill_(-1.0)
        self.target_prob = self.target_prob.to(self.device)

    def init_model(self,model):
        self.model = model

    def get_thresh_prob(self):
        source_label = self.source_dataset.labels
        cnt = Counter(source_label)
        for i in range(self.class_num):
            self.num_per_cls.append(cnt[i])

        _,max_num = cnt.most_common(1)[0]
        ep = 10

        for k in cnt:
            self.thresh_prob_class[k] = cnt[k] *1.0 /(max_num + ep)

    def get_perturb_point(self,input_source,labels_source):

        self.model.train(False)

        src_point = []
        tgt_point = []
        point_label = []
 
      
        for src_index,label in enumerate(labels_source):  
            if torch.rand(1) > self.thresh_prob_class[label.cpu().item()]:
                cond_one = self.target_label == label
                cond_two = self.target_prob > self.thresh_prob_pesudo
                cond = torch.bitwise_and(cond_one, cond_two)

                cond_index = torch.nonzero(cond,as_tuple=True)[0]

                if cond_index.size(0) > 0:
                    src_sample = input_source[src_index]
                    tgt_index = cond_index[torch.randint(cond_index.size(0),(1,))]

                    _,tgt_sample,_ = self.target_dataset[tgt_index]
                    
                    src_point.append(src_sample)
                    tgt_point.append(tgt_sample)
                    point_label.append(label)



        if len(point_label) <= 1:
            return None

        src_point = torch.stack(src_point)    
        tgt_point = torch.stack(tgt_point) 
        point_label = torch.as_tensor(point_label).long()

        src_point = src_point.to(self.device)
        tgt_point = tgt_point.to(self.device)
        point_label = point_label.to(self.device)

        perturb_num = src_point.size(0)
        cof = torch.rand(perturb_num,3,1,1,device=self.device)
        cof.requires_grad_(True)

        optim = SGD([cof],lr=0.001,momentum=0.9)

        loop = self.max_loop
        
        for i in range(loop):
            optim.zero_grad()

            perturbed_point = src_point + cof * (tgt_point - src_point)
            _,perturbed_output,_,_ = self.model(perturbed_point) 

            perturbed_output_softmax = 1 - F.softmax(perturbed_output, dim=1)
            perturbed_output_logsoftmax = torch.log(perturbed_output_softmax.clamp(min=self.epsilon))          
            loss = F.nll_loss(perturbed_output_logsoftmax, point_label,reduction='none')
            final_loss = torch.sum(loss)
            final_loss.backward()

            optim.step()
            cof.data.clamp_(0,1)
            self.model.zero_grad()

      
        cof = cof.detach()

        perturbed_point = src_point + cof * (tgt_point - src_point)
        
        self.model.train(True)
        return (perturbed_point,point_label)
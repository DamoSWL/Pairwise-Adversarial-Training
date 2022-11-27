


import torch.nn as nn
import backbone as backbone
from abstract import AbstractModel
import torch.nn.functional as F
import torch
import numpy as np
from utils import DIST



class GradientReverseLayer(torch.autograd.Function):
    iter_num = 0

    @staticmethod
    def forward(ctx, input):
        GradientReverseLayer.iter_num += 1
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        coeff = np.float(
            2.0 * (GradientReverseLayer.high_value - GradientReverseLayer.low_value) /
            (1.0 + np.exp(-GradientReverseLayer.alpha * GradientReverseLayer.iter_num / GradientReverseLayer.max_iter))
            - (GradientReverseLayer.high_value - GradientReverseLayer.low_value) + GradientReverseLayer.low_value
        )
        return - coeff * grad_output


class MDDNet(nn.Module, AbstractModel):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, classifier_width=1024,
                 class_num=31, classifier_depth=2, name='default', disable_dropout=False, grl_config=None,
                 freeze_backbone=False, normalize_features=False):
        AbstractModel.__init__(self, name)
        nn.Module.__init__(self)
        self.class_num = class_num
        self.bottleneck_dim = bottleneck_dim
        self.freeze_backbone = freeze_backbone
        self.normalize_features = normalize_features

        self.base_network = backbone.network_dict[base_net]()


        # init GradientReverseLayer parameterGradientReverseLayers
        GradientReverseLayer.alpha = 1.0
        GradientReverseLayer.low_value = 0
        GradientReverseLayer.high_value = 0.1
        GradientReverseLayer.max_iter = 1000

        self.use_bottleneck = use_bottleneck
        self.create_bottleneck_layer(use_dropout=not disable_dropout)

        self.create_f_and_fhat_classifiers(
            bottleneck_dim, classifier_width, class_num, classifier_depth, use_dropout=not disable_dropout)

        self.softmax = nn.Softmax(dim=1)

        # collect parameters
        self.parameter_list = [{"params": self.base_network.parameters(), 'lr_scale': 0.1},
                               {"params": self.bottleneck_layer.parameters(), 'lr_scale': 1},
                               {"params": self.classifier_layer.parameters(), 'lr_scale': 1},
                               {"params": self.classifier_layer_2.parameters(), 'lr_scale': 1}]

    def create_bottleneck_layer(self, use_dropout):
        bottleneck_layer_list = [
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.BatchNorm1d(self.bottleneck_dim),
            nn.ReLU()
        ]
        if use_dropout is True:
            bottleneck_layer_list.append(nn.Dropout(0.5))

        self.bottleneck_layer = nn.Sequential(*bottleneck_layer_list)

        # init
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)

    def create_f_and_fhat_classifiers(self, bottleneck_dim, classifier_width, class_num, classifier_depth,
                                      use_dropout=True):
        self.classifier_layer = self.create_classifier(
            bottleneck_dim, classifier_width, class_num, classifier_depth, use_dropout=use_dropout)
        self.classifier_layer_2 = self.create_classifier(
            bottleneck_dim, classifier_width, class_num, classifier_depth, use_dropout=use_dropout)
        self.initialize_classifiers()

    def create_classifier(self, bottleneck_dim, width, class_num, depth=2, use_dropout=True):
        layer_list = []
        input_size = bottleneck_dim
        for ith_layer in range(depth - 1):
            layer_list.append(nn.Linear(input_size, width))

            layer_list.append(nn.ReLU())

            if use_dropout is True:
                layer_list.append(nn.Dropout(0.5))

            input_size = width

        layer_list.append(nn.Linear(width, class_num))
        classifier = nn.Sequential(*layer_list)
        return classifier

    def forward(self, inputs):
        features = self.feature_forward(inputs)
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        # gradient reversal layer helps fuse the minimax problem into one loss function
        features_adv = GradientReverseLayer.apply(features)
        outputs_adv = self.classifier_layer_2(features_adv)

        return features, outputs, softmax_outputs, outputs_adv

    def feature_forward(self, inputs):
        if self.freeze_backbone is True:
            with torch.no_grad():
                features = self.base_network(inputs)
        else:
            features = self.base_network(inputs)

        if self.use_bottleneck:
            features = self.bottleneck_layer(features)

        if self.normalize_features is True:
            features_norm = torch.norm(features, p=2, dim=1).detach()
            features = features / features_norm.unsqueeze(1)
        return features

    def logits_forward(self, inputs):
        features = self.feature_forward(inputs)
        logits = self.classifier_layer(features)
        return logits

    def initialize_classifiers(self):
        self.xavier_initialization(self.classifier_layer)
        self.xavier_initialization(self.classifier_layer_2)

    def xavier_initialization(self, layers):
        for layer in layers:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(0.0)

    def initialize_bottleneck(self):
        for b_layer in self.bottleneck_layer:
            if type(b_layer) == nn.Linear:
                torch.nn.init.xavier_normal_(b_layer.weight)
                b_layer.bias.data.fill_(0.0)


class MDD(object):
    def __init__(self, base_net='ResNet50', classifier_width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=2,
                 classifier_depth=2, name='default_MDD', bottleneck_dim=2048,
                 disable_dropout=False, grl_config=None,
                 freeze_backbone=False, args=None):
        self.c_net = MDDNet(base_net, use_bottleneck=use_bottleneck, classifier_width=classifier_width,
                            class_num=class_num, name=name, 
                            classifier_depth=classifier_depth, bottleneck_dim=bottleneck_dim,
                            disable_dropout=disable_dropout, grl_config=grl_config,
                            freeze_backbone=freeze_backbone, normalize_features=args['normalize_features'])

        self.device = torch.device(args['gpu'] if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:          
            self.c_net = self.c_net.to(self.device)
            
        self.srcweight = srcweight
        self.epsilon = 1e-7
        self.step_num = 0
        
        self.s_centroid_flag = torch.zeros(self.class_num).bool()
        self.t_centroid_flag = torch.zeros(self.class_num).bool()

        # if self.args.proto_loss is True:
        self.s_centroid = torch.zeros(self.class_num, bottleneck_dim).to(self.device)
        self.t_centroid = torch.zeros(self.class_num, bottleneck_dim).to(self.device)
        self.align_loss_function = DIST() #nn.MSELoss()

        self.pat = None

    def setPAT(self,pat):
        self.pat = pat

    def get_loss(self, inputs, labels_source, tgt_index):  # inputs is a concatenation of source and target data
       
        class_criterion = nn.CrossEntropyLoss().to(self.device)

        input_source = inputs.narrow(0, 0, labels_source.size(0))

        features_both, outputs, _, outputs_adv = self.c_net(inputs)

        # split source and target features and outputs from the concatenated data
        features_src = features_both.narrow(0, 0, labels_source.size(0))
        features_tgt = features_both.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        # f(x)
        outputs_src = outputs.narrow(0, 0, labels_source.size(0))
        label_preds_src = outputs_src.max(1)[1]
        outputs_tgt = outputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        probs_tgt = F.softmax(outputs_tgt, dim=1)


        #obtain the target pseudo label and prob
        tgt_prob,tgt_label = torch.max(probs_tgt,dim=1)
        self.pat.target_label[tgt_index] = tgt_label
        self.pat.target_prob[tgt_index] = tgt_prob.data


        # f'(x)
        outputs_adv_src = outputs_adv.narrow(0, 0, labels_source.size(0))
        outputs_adv_tgt = outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))


        # classification loss on source domain
        if self.args['mask_classifier'] is True:
            outputs_src_masked, _, _ = self.mask_clf_outputs(
                outputs_src, outputs_adv_src, outputs_adv_tgt, labels_source)
            classifier_loss = class_criterion(outputs_src_masked, labels_source)
        else:
            classifier_loss = class_criterion(outputs_src, labels_source)


        if self.args['mask_divergence'] is True:
            outputs_src, outputs_adv_src, outputs_adv_tgt = self.mask_clf_outputs(
                outputs_src, outputs_adv_src, outputs_adv_tgt, labels_source)

        # use $f$ as the target for $f'$
        target_adv = outputs.max(1)[1]  # categorical labels from $f$
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        # source classification acc
        classifier_acc = (label_preds_src == labels_source).sum().float() / labels_source.size(0)

        # adversarial loss for source domain
        classifier_loss_adv_src = class_criterion(outputs_adv_src, target_adv_src)

        # adversarial loss for target domain, opposite sign with source domain
        prob_adv = 1 - F.softmax(outputs_adv_tgt, dim=1)
        prob_adv = prob_adv.clamp(min=self.epsilon)
        logloss_tgt = torch.log(prob_adv)
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        # total adversarial loss
        adv_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt
   

        # adversarial traning loss
        adverersial_traning_loss = 0
        perturbed_data = self.pat.get_perturb_point(input_source,labels_source)
        if perturbed_data is not None:
            perturbed_point = perturbed_data[0]
            point_label = perturbed_data[1]
            _,perturbed_output,_,_ = self.c_net(perturbed_point) 
            adverersial_traning_loss = class_criterion(perturbed_output,point_label)
            adverersial_traning_loss *= self.pat.pat_cof 

        # loss for explicit alignment
        explicit_alignment_loss = 0
        if self.args['align_loss'] is True:
            explicit_alignment_loss = self.get_explicit_alignment_loss(features_src, features_tgt, labels_source, probs_tgt)
            explicit_alignment_loss *= self.args['align_weight']


        total_loss = classifier_loss + adv_loss + explicit_alignment_loss + adverersial_traning_loss
       
        self.iter_num += 1

        train_losses = {
            'total_loss': total_loss,
            'classifier_loss': classifier_loss,
            'adv_loss': adv_loss,  
            'explicit_alignment_loss': explicit_alignment_loss,
            'adverersial_traning_loss':adverersial_traning_loss
        }


        return train_losses

    def get_explicit_alignment_loss(self, s_feature, t_feature, s_labels, t_probs):
        
        _, d = s_feature.shape

        s_feature = F.normalize(s_feature,dim=1)
        t_feature = F.normalize(t_feature,dim=1)


        # get labels
        t_prob, t_labels = torch.max(t_probs, dim=1)

        # count number of examples in each class
        zeros = torch.zeros(self.class_num).to(self.device)
        s_n_classes_raw = zeros.scatter_add(
            0,
            s_labels,
            torch.ones_like(s_labels, dtype=torch.float)).to(self.device)
        t_n_classes_raw = zeros.scatter_add(
            0,
            t_labels,
            torch.ones_like(t_labels, dtype=torch.float)).to(self.device)

        # image count cannot be 0, when calculating centroids
        ones = torch.ones_like(s_n_classes_raw).to(self.device)
        s_n_classes = torch.maximum(s_n_classes_raw, ones)
        t_n_classes = torch.maximum(t_n_classes_raw, ones)

        # calculating centroids, sum and divide
        zeros = torch.zeros(self.class_num, d).to(self.device)
        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), t_feature)
        s_centroid = torch.div(s_sum_feature, s_n_classes.view(self.class_num, 1))
        t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.class_num, 1))

        # Moving Centroid
        decay = self.args['align_decay_rate']
        if self.args['moving_centroid'] is True:
            for k in range(self.class_num):
                if s_n_classes_raw[k] > 0:
                    if self.s_centroid_flag[k] == True:             
                        s_centroid[k,:] = (1 - decay) * self.s_centroid[k,:] + decay * s_centroid[k,:]
                else:
                    s_centroid[k,:] =  self.s_centroid[k,:] + 0 * s_centroid[k,:]

                if t_n_classes_raw[k] > 0:
                    if self.t_centroid_flag[k] == True: 
                        t_centroid[k,:] = (1 - decay) * self.t_centroid[k,:] + decay * t_centroid[k,:]
                else:
                    t_centroid[k,:] =  self.t_centroid[k,:] + 0 * t_centroid[k,:]

        semantic_loss = self.align_loss_function.get_dist(s_centroid, t_centroid)
        semantic_loss = torch.sum(semantic_loss)

        self.s_centroid = s_centroid.detach()
        self.t_centroid = t_centroid.detach()


        for k in range(self.class_num):
            if (s_n_classes_raw[k] > 0) and (self.s_centroid_flag[k] == False):
                self.s_centroid_flag[k] = True              
            if (t_n_classes_raw[k] > 0) and (self.t_centroid_flag[k] == False):
                self.t_centroid_flag[k] = True

        return semantic_loss


    def mask_clf_outputs(self, outputs_src, outputs_adv_src, outputs_adv_tgt, labels_source):
        mask = torch.zeros(outputs_src.shape[1])
        mask[labels_source.unique()] = 1
        mask = mask.repeat((outputs_src.shape[0], 1)).to(self.device)
        outputs_src = outputs_src * mask
        outputs_adv_src = outputs_adv_src * mask

        outputs_adv_tgt = outputs_adv_tgt * mask
        return outputs_src, outputs_adv_src, outputs_adv_tgt

    def predict(self, inputs):
        _, _, softmax_outputs, _ = self.c_net(inputs)
        return softmax_outputs


    def get_features(self, inputs):
        features, _, _, _ = self.c_net(inputs)
        return features

    def get_parameter_list(self):
        c_net_params = self.c_net.parameter_list
        return c_net_params

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

    def copy_params_from(self, source):
        for this_param, source_param in zip(self.c_net.parameters(), source.c_net.parameters()):
            this_param.data = source_param.detach().clone()

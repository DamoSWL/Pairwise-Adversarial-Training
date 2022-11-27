import argparse
import torch 
import torch.cuda
import torch.optim as optim
from optim import INVScheduler
from utils import *
import numpy as np
from MDD import MDD
import torch.optim as optim
from time import time
import time
import yaml
from PAT import PAT
from dataloader import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='./config/office31.yml')
    args = parser.parse_args()

    set_global_random_seed(47)

    with open(args.config,'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    g_device = torch.device(cfg['gpu'] if torch.cuda.is_available() else 'cpu')
  
    train_source_loader, train_source_sampler = GetDataLoader(cfg,
                                                        cfg['src_file'], 
                                                        batch_size=cfg['batch_size'],
                                                        dataset_dir=cfg['dataset_dir'],
                                                        sample_mode_with_ground_truth_labels=cfg['source_sample_mode'],
                                                        is_source=True)


    train_target_loader, _ = GetDataLoader(cfg,
                                            cfg['tgt_file'], 
                                            batch_size=cfg['batch_size'],
                                            dataset_dir=cfg['dataset_dir'],
                                            sample_mode_with_ground_truth_labels=False,
                                            drop_flag=True,
                                            is_source=False)


    test_target_loader, _= GetDataLoader(cfg,
                                        cfg['tgt_file_eval'], 
                                        batch_size=cfg['batch_size'], 
                                        dataset_dir=cfg['dataset_dir'],
                                        is_train=False, 
                                        is_source=False)

                     
    model = MDD(classifier_width=cfg['feature_size'],class_num=cfg['class_num'],srcweight=cfg['srcweight'],args=cfg)
    pat = PAT(cfg)

    tgt_number = len(train_target_loader.dataset)
    pat.init_model(model.c_net)
    pat.init_target_pseudo(tgt_number)
    pat.init_dataset(train_source_loader.dataset,train_target_loader.dataset)

    model.setPAT(pat)

    param_groups = model.get_parameter_list()
    optimizer = optim.SGD(param_groups, lr=0.001, momentum=0.9, weight_decay=0.0005,nesterov=True) 
    lr_scheduler = INVScheduler(gamma=0.001,decay_rate=0.75,init_lr=0.001)          

            
    best_acc = 0
    best_balance_acc = 0
    best_class_acc = []
     

    src_loader_iter = iter(train_source_loader)
    tgt_loader_iter = iter(train_target_loader)

    print(cfg)
    print('start training...')
    total_time = 0
    
    for current_iter in range(cfg['max_iter']):
        start_time = time.time()

        model.set_train(True)

        optimizer = lr_scheduler.next_optimizer(optimizer,current_iter // 5)

        if current_iter % len(train_source_loader) == 0:
            src_loader_iter = iter(train_source_loader)
                
        if current_iter % len(train_target_loader) == 0:
            tgt_loader_iter = iter(train_target_loader)
            
            
        _,src_image,src_label = src_loader_iter.next()
        tgt_index,tgt_image,_ = tgt_loader_iter.next()  

        src_image = src_image.to(g_device)
        src_label = src_label.to(g_device)
        tgt_image = tgt_image.to(g_device)
        tgt_index = tgt_index.to(g_device)

        optimizer.zero_grad()


        inputs = torch.cat((src_image, tgt_image), dim=0)
        total_loss = model.get_loss(inputs,src_label, tgt_index)
        total_loss['total_loss'].backward()

        optimizer.step()
            
        end_time = time.time()
        total_time += end_time-start_time

        if (current_iter+1) % 50 == 0:
            print('time_consumed: ',total_time)
            print('iteration {} ,total_loss:{:.4f} classifier_loss {:.4f} adv_loss: {:.4f} explicit_alignment_loss: {:.4f} adverersial_traning_loss: {:.4f}'.format(
                current_iter+1,total_loss['total_loss'],total_loss['classifier_loss'],total_loss['adv_loss'],total_loss['explicit_alignment_loss'],total_loss['adverersial_traning_loss']))

        with open(cfg['output'],'a') as f:
            f.write('iteration {} ,total_loss:{:.4f} classifier_loss {:.4f} adv_loss: {:.4f} explicit_alignment_loss: {:.4f} adverersial_traning_loss: {:.4f}\n'.format(
                current_iter+1,total_loss['total_loss'],total_loss['classifier_loss'],total_loss['adv_loss'],total_loss['explicit_alignment_loss'],total_loss['adverersial_traning_loss']))     

        if (current_iter+1) % cfg['test_interval'] == 0:
            tgt_pred_label = []
            tgt_ground_label = []
            prob = []

            model.set_train(False)

            with torch.no_grad():

                for _,test_images,test_labels in test_target_loader:
                   
                    test_images = test_images.to(g_device)
                    test_labels = test_labels.to(g_device)
            
                    
                    output_test = model.predict(test_images)

                    _,predict_test = torch.max(output_test, dim=1)
                    predict_test = torch.squeeze(predict_test).cpu()
                    tgt_pred_label += predict_test.numpy().tolist()
                    tgt_ground_label += test_labels.cpu().numpy().tolist()
                
            

                tgt_pred_label = torch.as_tensor(tgt_pred_label)
                tgt_ground_label = torch.as_tensor(tgt_ground_label)

            
                tgt_acc = accuracy(tgt_pred_label,tgt_ground_label)
                tgt_balance_acc = balance_accuracy(tgt_pred_label,tgt_ground_label,cfg['class_num'])
                class_acc = class_accuracy(tgt_pred_label,tgt_ground_label,cfg['class_num'])
            
                if tgt_balance_acc > best_balance_acc:
                    best_acc = tgt_acc
                    best_balance_acc = tgt_balance_acc
                    best_class_acc = class_acc

                    print('iteration {} best_acc:{:.2f} best_balance_acc {:.2f}'.format(current_iter+1, best_acc,best_balance_acc))

                    with open(cfg['output'],'a') as f:
                        f.write('iteration {} best_acc:{:.2f} best_balance_acc {:.2f}\n'.format(current_iter+1, best_acc,best_balance_acc))
    
    print('training finished')
    print('best_acc:{:.2f} best_balance_acc {:.2f}'.format(best_acc,best_balance_acc))
    print(best_class_acc)
    print('total time: ',total_time)

    with open(cfg['output'],'a') as f:
        f.write('best_acc:{:.2f} best_balance_acc {:.2f}\n'.format(best_acc,best_balance_acc))
        f.write(best_class_acc)
  


 

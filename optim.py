import torch.optim as optim

def generate_optimizer(networks):  
    lr = 1
    for name, network in networks.items():
        param_list = []
        if name == 'C':
            param_list.append({"params": network.parameters(), "lr_scale": 10*lr})
        elif name == 'G':
            param_list.append({"params": network.parameters(), "lr_scale": lr})
 
        
    optimizer = optim.SGD(param_list, lr=0.001, momentum=0.9, weight_decay=0.0005)      
    return optimizer

def inv_lr_scheduler(optimizer, iter_num, alpha=0.001, power=0.75, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + alpha * iter_num) ** (- power)
   
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_scale']      
    return optimizer

class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_scale']
        return optimizer
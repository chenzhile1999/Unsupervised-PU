
import torch
from torch import optim as optim
from torch.optim import lr_scheduler
import os

def get_Optimizer(opt, model):
    if opt.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': opt.lr}], lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': opt.lr}], lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    else:
        raise Exception('Unexpected Optimizer of {}'.format(opt.optimizer))
    return optimizer

def get_Scheduler(opt, optimizer):
    if opt.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.gamma, last_epoch=-1 if opt.start_epoch==0 else opt.start_epoch, verbose=True)
    elif opt.scheduler == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200], gamma=opt.gamma, last_epoch=-1 if opt.start_epoch==0 else opt.start_epoch, verbose=True)
    elif opt.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-4, last_epoch=-1 if opt.start_epoch==0 else opt.start_epoch, verbose=True)
    elif opt.scheduler == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma, verbose=True)
    elif opt.scheduler == 'cycle':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=opt.lr, step_size_up=50, step_size_down=50,
                                                      mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                      cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1 if opt.start_epoch==0 else opt.start_epoch, verbose=True)
    else:
        raise Exception('Unexpected Scheduler of {}'.format(opt.scheduler))
    return scheduler

class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def write_args(self, log_name):
        params_dict = vars(self.__args)

        log_dir = os.path.join('./%s' % (params_dict['log_dir']))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        args_name = os.path.join(log_dir, '%s_args.txt' % (log_name))

        with open(args_name, 'wt') as args_fid:
            args_fid.write('----' * 10 + '\n')
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
            args_fid.write('----' * 10 + '\n')
            for k, v in sorted(params_dict.items()):
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
            args_fid.write('----' * 10 + '\n')

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)

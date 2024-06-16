

import random
import numpy as np
import torch


def img_param_init(args):
    dataset = args.dataset
    if dataset =='BrainTumor':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='havior':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='RealSkin':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='Dermnet':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='OfficeHome':
        domains = ['Art','Clipart','RealWorld', 'Product' ]
    if dataset == 'ModernOffice31':
        domains = ['a','d','s','w']
    if dataset =='PACS':
        domains = ['C', 'P', 'S', 'A']
    args.domains = domains
    if args.dataset =='BrainTumor':
        args.num_classes = 4
    if args.dataset =='havior':
        args.num_classes = 23
    if args.dataset =='RealSkin':
        args.num_classes = 7
    if args.dataset =='OfficeHome':
        args.num_classes = 65
    if args.dataset == 'ModernOffice31':
        args.num_classes = 31
    if args.dataset =='Dermnet':
        args.num_classes = 23
    return args


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

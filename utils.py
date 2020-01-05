import torch
import torch, torch.nn as nn, torch.nn.functional as F
from load_data import get_dataset, get_dataset_test
from model import FraudNet
import sys
from glob import glob
from metrics import calc_metrics_classification
import numpy as np
from tensorboardX import SummaryWriter

def load_fraudnet(ch=5):

    net = FraudNet(ch=ch).to(device='cuda')
    path = './best_model/smote_sameWt_categorical_fc_0.3_5_0.194_0.194.th'
    net.load_state_dict(torch.load(path))
    return net.to(device='cuda').eval()

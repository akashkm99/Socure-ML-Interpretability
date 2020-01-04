import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
# pip install shap
import shap

from load_data import get_dataset, get_dataset_test

model = torch.load('./saved_checkpoints/continuous_sameWt_eqSample_new_fc_0.3_2_0.333_0.333.th', map_location='cpu')
train_loader, valid_loader, test_loader = get_dataset(minibatch_size=128)

# since shuffle=True, this is a random sample of test data
batch = next(iter(test_loader))
images, _ = batch

background = images[:100]
test_images = images[100:103]

e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_images)

print (shap_values.shape)

import os

import configs
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import common
from segnet import SegNet
from PIL import Image
from torch import nn
from torch.nn import BatchNorm2d
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.utils import save_image

from Batchtransfer_EMA import BatchTransNorm


def get_batch(b_size, datasetpath=configs.EuroSAT_path, image_size=224):
    transform = common.TransformLoader(
    image_size).get_composed_transform(aug=False)
    transform_test = common.TransformLoader(
    image_size).get_composed_transform(aug=False)

    loader = torch.utils.data.DataLoader(common.SimpleDataset(transform, datasetpath=datasetpath), batch_size=b_size,
                                                  num_workers=0,
                                                  shuffle=True, drop_last=True)
    return next(iter(loader))[0]



model = SegNet(normlayer=nn.BatchNorm2d)
model.load_state_dict(torch.load('lab/autoencoder/model.pth', map_location=configs.device))
model.to(configs.device)
model.eval()


model_btrans = SegNet(normlayer=BatchTransNorm)
model_btrans.load_state_dict(torch.load('lab/autoencoder/model.pth', map_location=configs.device))
model_btrans.to(configs.device)
model_btrans.eval()


image_size = 300

content = get_batch(b_size=4, datasetpath="D:/downloaded_DS/fairface-img-margin025-trainval/val/", image_size=image_size)
_ = model_btrans(content)
    
out = model(content)
out_btras = model_btrans(content)


columns = 4      
rows = 2
fig = plt.figure(figsize=(columns*5, rows*5))
for i in range(columns):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(out[i].detach().permute(1, 2, 0))
    plt.axis('off')
    
    fig.add_subplot(rows, columns, i + columns + 1)
    plt.imshow(out_btras[i].detach().permute(1, 2, 0))
    plt.axis('off')
    
plt.show()

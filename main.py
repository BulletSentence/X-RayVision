import os
import shutil
import random
import torch
import torchvision
import numpy as np
import matplotlib.pyplot

from PIL import Image

torch.manual_seed(0)

print('Using PyTorch v', torch.__version__)

class_names = ['normal', 'sick', 'covid']
root_dir = 'Covid Database'
source_dirs = ['NORMAL', 'Viral Pneumonia', 'COVID-19']

if os.path.isdir(os.path.join((root_dir, source_dirs[1]))):
    os.mkdir(os.path.join(root_dir, 'test'))

    for i, d in enumerate(source_dirs):
        os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))

    for c in class_names:
        os.mkdir(os.path.join(root_dir, 'test', c))

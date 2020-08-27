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

class_name = ['normal', 'sick', 'covid']
root_dir = 'Covid Database'
source_dirs = ['NORMAL', 'Viral Pneumonia', 'COVID-19']

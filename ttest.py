import argparse
import os
import numpy as np


import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import sys

import matplotlib.pyplot as plt
import time


writer = SummaryWriter()

for i in range(10000):
	writer.add_scalar('g_loss', i, global_step=i, walltime=None)
	
	

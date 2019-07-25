import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append("../")  # ../../GAN-SDPC/

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--model_path", type=str, default='models/last_G.pt',
                    help="Chemin vers le générateur à charger")
parser.add_argument("-s", "--seed_path", type=str, default='seeds.txt',
                    help="Chemin vers le fichier contenant les seeds à générer.")
parser.add_argument("-r", "--results_path", type=str, default='results',
                    help="Nom du fichier contenant les résultats")
parser.add_argument("--eps", type=float, default=0.5, help="batchnorm: espilon for numerical stability")
parser.add_argument("--lrelu", type=float, default=0.2, help="LeakyReLU : alpha")
parser.add_argument("--latent_dim", type=int, default=6, help="dimensionality of the latent space")
parser.add_argument("-i", "--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Afficher des informations complémentaire.")
opt = parser.parse_args()
print(opt)

# Dossier de sauvegarde
os.makedirs(opt.results_path, exist_ok=True)

# Initialize generator 
NL = nn.LeakyReLU(opt.lrelu, inplace=True)
opts_conv = dict(kernel_size=5, stride=2, padding=2, padding_mode='zeros')
channels = [64, 128, 256, 512]
class Generator(nn.Module):
    def __init__(self, verbose=opt.verbose):
        super(Generator, self).__init__()
        
        def generator_block(in_filters, out_filters):
            block = [nn.UpsamplingNearest2d(scale_factor=opts_conv['stride']), nn.Conv2d(in_filters, out_filters, kernel_size=opts_conv['kernel_size'], stride=1, padding=opts_conv['padding'], padding_mode=opts_conv['padding_mode']), nn.BatchNorm2d(out_filters, opt.eps), NL]

            return block

        self.verbose = verbose
        self.init_size = opt.img_size // opts_conv['stride']**3
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, channels[3] * self.init_size ** 2), NL)


        self.conv1 = nn.Sequential(*generator_block(channels[3], channels[2]),)
        self.conv2 = nn.Sequential(*generator_block(channels[2], channels[1]),)
        self.conv3 = nn.Sequential(*generator_block(channels[1], channels[0]),)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(channels[0], opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, z):
        if self.verbose: print("G")
        # Dim : opt.latent_dim
        out = self.l1(z)
        if self.verbose: print("l1 out : ",out.shape)
        out = out.view(out.shape[0], channels[3], self.init_size, self.init_size)
        # Dim : (channels[3], opt.img_size/8, opt.img_size/8)
        if self.verbose: print("View out : ",out.shape)

        out = self.conv1(out)
        # Dim : (channels[3]/2, opt.img_size/4, opt.img_size/4)
        if self.verbose: print("Conv1 out : ",out.shape)
        out = self.conv2(out)
        # Dim : (channels[3]/4, opt.img_size/2, opt.img_size/2)
        if self.verbose: print("Conv2 out : ",out.shape)
        out = self.conv3(out)
        # Dim : (channels[3]/8, opt.img_size, opt.img_size)
        if self.verbose: print("Conv3 out : ",out.shape)
        
        img = self.conv_blocks(out)
        # Dim : (opt.chanels, opt.img_size, opt.img_size)
        if self.verbose: print("img out : ", img.shape)

        return img

    def _name(self):
        return "Generator"
generator = Generator()
print_network(generator)

# Chargement
load_model(generator, None, opt.model_path)

# Lecture des seeds
with open(opt.seed_path, "r") as f:
    text = f.read().splitlines()
    
    params = []
    for line in text:
        line = line.replace(',','').split()
        print(line)
        params.append(line)
    
    #print(params)
    
# GPU paramétrisation
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Génération
params = Variable(Tensor(np.asarray(params,dtype=np.float64)))
sampling(params, generator, opt.results_path, 0)
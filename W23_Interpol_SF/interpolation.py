import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter

sys.path.append("../")  # ../../GAN-SDPC/

from utils import *

#########
# python use_G.py -p scan7_models_eps0.1lrelu1e06/last_G.pt -s seed_dataset_kdc.txt --eps 0.1 --lrelu 0.000001 --GPU 1
# python use_G.py -p models_eps0.0lrelu0.01/last_G.pt -s seed_dataset_kdc.txt --eps 0.0 --lrelu 0.01 --GPU 1 --kernels_size 9 --padding 4
#########

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--model_path", type=str, default='models/last_G.pt',
                    help="Chemin vers le générateur à charger")
parser.add_argument("-s", "--seed_path", type=str, default='seeds.txt',
                    help="Chemin vers le fichier contenant les seeds à générer.")
parser.add_argument("-r", "--results_path", type=str, default='results',
                    help="Dossier contenant les résultats")
parser.add_argument("-t", "--tag", type=str, default='image',
                    help="Nom du fichier contenant les résultats")
parser.add_argument("--eps", type=float, default=0.5, help="batchnorm: espilon for numerical stability")
parser.add_argument("--lrelu", type=float, default=1e-06, help="LeakyReLU : alpha")
parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent space")
parser.add_argument("--kernels_size", type=int, default=9, help="Taille des kernels")
parser.add_argument("--padding", type=int, default=4, help="Taille du padding")
parser.add_argument("-i", "--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--points", type=int, default=5, help="number of inter points between interpolation")
parser.add_argument("--sample", type=int, default=3, help="number of interpolation")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Afficher des informations complémentaire.")
parser.add_argument("--runs_path", type=str, default='Interpol_SF/inter/',
                    help="Dossier de stockage des résultats sous la forme : Experience_names/parameters/")
parser.add_argument("--GPU", type=int, default=0, help="Identifiant du GPU à utiliser.")
opt = parser.parse_args()
print(opt)

# Dossier de sauvegarde
os.makedirs(opt.results_path, exist_ok=True)

# Gestion du time tag
try:
    time_tag = datetime.datetime.now().isoformat(sep='_', timespec='seconds')
except TypeError:
    # Python 3.5 and below
    # 'timespec' is an invalid keyword argument for this function
    time_tag = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_')
time_tag = time_tag.replace(':','.')

# ----------
# Initialize generator 
# ----------
NL = nn.LeakyReLU(opt.lrelu, inplace=True)
opts_conv = dict(kernel_size=opt.kernels_size, stride=2, padding=opt.padding, padding_mode='zeros')
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

class Discriminator(nn.Module):
    def __init__(self,verbose=opt.verbose):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, **opts_conv), NL]#, nn.Dropout2d(0.25)
            if bn:
                block.append(nn.BatchNorm2d(out_filters, opt.eps))
            return block

        self.verbose = verbose

        self.conv1 = nn.Sequential(*discriminator_block(opt.channels, channels[0], bn=False),)
        self.conv2 = nn.Sequential(*discriminator_block(channels[0], channels[1]),)
        self.conv3 = nn.Sequential(*discriminator_block(channels[1], channels[2]),)
        self.conv4 = nn.Sequential(*discriminator_block(channels[2], channels[3]),)

        # The height and width of downsampled image
        self.init_size = opt.img_size // opts_conv['stride']**4
        self.adv_layer = nn.Sequential(nn.Linear(channels[3] * self.init_size ** 2, 1))#, nn.Sigmoid()

    def forward(self, img):
        if self.verbose:
            print("D")
            print("Image shape : ",img.shape)
            out = self.conv1(img)
            print("Conv1 out : ",out.shape)
            out = self.conv2(out)
            print("Conv2 out : ",out.shape)
            out = self.conv3(out)
            print("Conv3 out : ",out.shape)
            out = self.conv4(out)
            print("Conv4 out : ",out.shape)

            out = out.view(out.shape[0], -1)
            print("View out : ",out.shape)
            validity = self.adv_layer(out)
            print("Val out : ",validity.shape)
        else:
            # Dim : (opt.chanels, opt.img_size, opt.img_size)
            out = self.conv1(img)
            # Dim : (channels[3]/8, opt.img_size/2, opt.img_size/2)
            out = self.conv2(out)
            # Dim : (channels[3]/4, opt.img_size/4, opt.img_size/4)
            out = self.conv3(out)
            # Dim : (channels[3]/2, opt.img_size/4, opt.img_size/4)
            out = self.conv4(out)
            # Dim : (channels[3], opt.img_size/8, opt.img_size/8)

            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)
            # Dim : (1)

        return validity

    def _name(self):
        return "Discriminator"
generator = Generator()

print_network(generator)

load_model(generator, None, opt.model_path)

# ----------
# GPU paramétrisation
# ----------
cuda = True if torch.cuda.is_available() else False
if cuda:
    if torch.cuda.device_count() > opt.GPU: 
        torch.cuda.set_device(opt.GPU)
    generator.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Tensorboard
# ----------
path_data1 = "../runs/" + opt.runs_path
path_data2 = "../runs/" + opt.runs_path + time_tag[:-1] + "/"

# Les runs sont sauvegarder dans un dossiers "runs" à la racine du projet, dans un sous dossiers opt.runs_path.
os.makedirs(path_data1, exist_ok=True)
os.makedirs(path_data2, exist_ok=True)

writer = SummaryWriter(log_dir=path_data2)

# ----------
# Interpolation noise
# ----------
N = opt.sample
points = opt.points

# Select 2 points
ans = "n"
while ans != 'y':
	a = np.random.normal(0, 1, (N, opt.latent_dim))
	tensorboard_sampling(Variable(Tensor(a)), generator, writer, 0, image_type="Point A")
	print("Le point tiré convient-il ? (y/n)")
	ans = input()
ans = "n"
while ans != 'y':
	b = np.random.normal(0, 1, (N, opt.latent_dim))
	tensorboard_sampling(Variable(Tensor(b)), generator, writer, 0, image_type="Point B")
	print("Le point tiré convient-il ? (y/n)")
	ans = input()

diff = np.abs(a-b)

c = list()
for i in range(N):
    c.append(np.linspace(a[i], b[i], points, endpoint=True))
c = np.asarray(c).reshape((N*points,opt.latent_dim))
print(c)
print(c.shape)

# Génération
noise = Variable(Tensor(c))
sampling(noise, generator, opt.results_path, 0, tag=opt.tag, nrow=points)
tensorboard_sampling(noise, generator, writer, 0, nrow=points, image_type="Interpolation A vers B")
for i,line in enumerate(c):
	sampling(Variable(Tensor(line)), generator, opt.results_path, 0, tag=opt.tag+"_"+str(i), nrow=1)

# Analyse de l'espace
print(diff)
print("Mean diff : :\n",diff.mean(axis=1))
print("Std diff : :\n",diff.std(axis=1))

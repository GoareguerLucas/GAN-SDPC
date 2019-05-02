import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import sys
sys.path.append("../")#../../GAN-SDPC/

from SimpsonsDataset import SimpsonsDataset,FastSimpsonsDataset
from utils import *

import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lrD", type=float, default=0.0004, help="adam: learning rate for D")
parser.add_argument("--lrG", type=float, default=0.0004, help="adam: learning rate for G")
parser.add_argument("--eps", type=float, default=0.5, help="batchnorm: espilon for numerical stability")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=40, help="interval between image sampling")
parser.add_argument("--sample_path", type=str, default='images')
parser.add_argument("--model_save_interval", type=int, default=2500, help="interval between image sampling")
parser.add_argument('--model_save_path', type=str, default='models')
opt = parser.parse_args()
print(opt)

# Dossier de sauvegarde
#os.makedirs("images", exist_ok=True)
#os.makedirs("model", exist_ok=True)
os.makedirs(opt.sample_path, exist_ok=True)
os.makedirs(opt.model_save_path, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Couches Non-linaire  
#NONLIN = nn.ELU(True)
NONLIN = nn.LeakyReLU(0.2, inplace=True)
#NONLIN = nn.ReLU(True)

def weights_init_normal(m,factor=1.0):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        n=float(m.in_channels*m.kernel_size[0]*m.kernel_size[1])
        n+=float(m.kernel_size[0]*m.kernel_size[1]*m.out_channels)
        n=n/2.0
        m.weight.data.normal_(0,math.sqrt(factor/n))
        m.bias.data.zero_()
        #torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        n=float(m.in_features+m.out_features)
        n=n/2.0
        m.weight.data.normal_(0,math.sqrt(factor/n))
        m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
        #torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        #torch.nn.init.constant_(m.bias.data, 0.0)


class Generation(nn.Module):
    def __init__(self):
        super(Generation, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, opt.eps),
            NONLIN,
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, opt.eps),
            NONLIN,
            nn.Conv2d(64, opt.channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Analyse(nn.Module):
    def __init__(self):
        super(Analyse, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1), NONLIN]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, opt.eps))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size
        self.adv_layer = nn.Linear(128 * ds_size ** 2, opt.latent_dim)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        out = self.adv_layer(out)

        return out


# Loss function
adversarial_loss = torch.nn.L1Loss()
#pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generation = Generation()
analyse = Analyse()

if cuda:
    generation.cuda()
    analyse.cuda()
    adversarial_loss.cuda()
    #pixelwise_loss.cuda()

# Initialize weights
generation.apply(weights_init_normal)
analyse.apply(weights_init_normal)

# Configure data loader
dataloader = load_data("../../cropped/cp/",opt.img_size,opt.batch_size)

# Optimizers
optimizer_G = torch.optim.Adam(generation.parameters(), lr=opt.lrG, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(itertools.chain(analyse.parameters(), generation.parameters()), lr=opt.lrD, betas=(opt.b1, opt.b2))

#scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=1000, gamma=0.1)
#scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=1000, gamma=0.1)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

G_losses = []
D_losses = []
g_losses = []
d_losses = []
D_x = []
D_G_z = []
d_x = []
d_g_z = []

save_dot = 10 # Nombre d'epochs avant de sauvegarder un point des courbes
batch_on_save_dot = save_dot*len(dataloader)

t_total = time.time()
for epoch in range(1,opt.n_epochs+1):
    t_epoch = time.time()
    #scheduler_G.step()
    #scheduler_D.step()
    for i, (imgs, _) in enumerate(dataloader):
        t_batch = time.time()

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        
        # Generate a batch of images
        gen_imgs = generation(z)
        
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(generation(analyse(gen_imgs)), gen_imgs)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        
        # Measure discriminator's ability to classify real from generated samples        
        d_real = generation(analyse(real_imgs))
        d_fake = generation(analyse(gen_imgs.detach()))

        d_loss_real = adversarial_loss(d_real, real_imgs)
        d_loss_fake = adversarial_loss(d_fake, gen_imgs.detach())
        
        d_loss = (d_loss_real + d_loss_fake) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time: %fs]"
            % (epoch, opt.n_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item(), time.time()-t_batch)
        )
    
        # Save Losses and scores for plotting later
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
    # Save samples
    if epoch % opt.sample_interval == 0:
        save_image(gen_imgs.data[:25], "%s/%d.png" % (opt.sample_path, epoch), nrow=5, normalize=True)
    
    # Save Losses and scores for plotting later
    if epoch % save_dot == 0:
        G_losses.append(sum(g_losses)/batch_on_save_dot)
        D_losses.append(sum(d_losses)/batch_on_save_dot)
        g_losses = []
        d_losses = []
    
    # Save models
    if epoch % opt.model_save_interval == 0:
        num = str(int(epoch / opt.model_save_interval))
        save_model(analyse,optimizer_D,epoch,opt.model_save_path+"/"+num+"_D.pt")
        save_model(generation,optimizer_G,epoch,opt.model_save_path+"/"+num+"_G.pt")
    
    # Intermediate plot
    if epoch % (opt.n_epochs/4) == 0:
        #Plot losses            
        plot_losses(G_losses,D_losses)
    
    print("[Epoch Time: ",time.time()-t_epoch,"s]")

print("[Total Time: ",time.strftime("%Hh:%Mm:%Ss",time.gmtime(time.time()-t_total)),"]")

#Plot losses            
plot_losses(G_losses,D_losses)

# Save model for futur training
save_model(analyse,optimizer_D,epoch,opt.model_save_path+"/last_D.pt")
save_model(generation,optimizer_G,epoch,opt.model_save_path+"/last_G.pt")

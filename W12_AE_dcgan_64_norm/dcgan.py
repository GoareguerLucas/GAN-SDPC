import argparse
import os
import numpy as np

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

import time
import datetime
tag = datetime.datetime.now().isoformat(timespec='seconds') + '_'

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lrD", type=float, default=0.00001, help="adam: learning rate for D")
parser.add_argument("--lrG", type=float, default=0.0001, help="adam: learning rate for G")
parser.add_argument("--lrE", type=float, default=0.001, help="adam: learning rate for E")
parser.add_argument("--eps", type=float, default=0.00005, help="batchnorm: espilon for numerical stability")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("-i", "--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("-s", "--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--sample_path", type=str, default='images')
parser.add_argument("-d", "--depth", action="store_true", help="Utiliser si utils.py et SimpsonsDataset.py sont deux dossier au dessus.")
opt = parser.parse_args()
print(opt)

# Particular import
depth = ""
if opt.depth == True:
    depth = "../"
sys.path.append(depth+"../")#../../GAN-SDPC/

from SimpsonsDataset import SimpsonsDataset,FastSimpsonsDataset
from utils import *
from plot import *

# Dossier de sauvegarde
os.makedirs(opt.sample_path, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
NL = nn.LeakyReLU(0.2, inplace=True)
# (N + 2*p - k) / s +1 cf https://pytorch.org/docs/stable/nn.html#conv2d
opts_conv = dict(kernel_size=4, stride=2, padding=2, padding_mode='circular')
opts_conv = dict(kernel_size=9, stride=2, padding=4, padding_mode='circular')
opts_conv = dict(kernel_size=9, stride=2, padding=4, padding_mode='zeros')
verbose=False
# verbose=True

class Encoder(nn.Module):
    def __init__(self, verbose=verbose):
        super(Encoder, self).__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, **opts_conv), NL]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, opt.eps))
            return block

        self.verbose = verbose
        # use a different layer in the encoder using similarly max_filters
        self.max_filters = 512

        self.conv1 = nn.Sequential(*encoder_block(opt.channels, 64, bn=False),)
        self.conv2 = nn.Sequential(*encoder_block(64, 128),)
        self.conv3 = nn.Sequential(*encoder_block(128, 256),)
        self.conv4 = nn.Sequential(*encoder_block(256, self.max_filters),)

        self.init_size = opt.img_size // opts_conv['stride']**4
        self.vector = nn.Linear(self.max_filters * self.init_size ** 2, opt.latent_dim)
        # self.sigmoid = nn.Sequential(nn.Sigmoid(),)

    def forward(self, img):
        if self.verbose: print("Encoder")
        if self.verbose: print("Image shape : ",img.shape)
        out = self.conv1(img)
        if self.verbose: print("Conv1 out : ",out.shape)
        out = self.conv2(out)
        if self.verbose: print("Conv2 out : ",out.shape)
        out = self.conv3(out)
        if self.verbose: print("Conv3 out : ",out.shape)
        out = self.conv4(out)
        if self.verbose: print("Conv4 out : ",out.shape, " init_size=", self.init_size)

        out = out.view(out.shape[0], -1)
        if self.verbose: print("View out : ",out.shape)
        z = self.vector(out)
        # z = self.sigmoid(z)
        # if self.verbose: print("Z : ",z.shape)

        return z

class Generator(nn.Module):
    def __init__(self, verbose=verbose):
        super(Generator, self).__init__()

        # def generator_block(in_filters, out_filters):
        #     block = [nn.ConvTranspose2d(in_filters, out_filters, **opts_conv, output_padding=1), nn.BatchNorm2d(out_filters, opt.eps), NL]
        #
        #     return block
        def generator_block(in_filters, out_filters):
            block = [nn.UpsamplingNearest2d(scale_factor=opts_conv['stride']),            nn.Conv2d(in_filters, out_filters, kernel_size=opts_conv['kernel_size'], stride=1, padding=opts_conv['padding'], padding_mode=opts_conv['padding_mode']), nn.BatchNorm2d(out_filters, opt.eps), NL]

            return block

        self.verbose = verbose
        self.max_filters = 512
        self.init_size = opt.img_size // 8
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.max_filters * self.init_size ** 2), NL)


        self.conv1 = nn.Sequential(*generator_block(self.max_filters, 256),)
        self.conv2 = nn.Sequential(*generator_block(256, 128),)
        self.conv3 = nn.Sequential(*generator_block(128, 64),)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        if self.verbose: print("G")
        # Dim : opt.latent_dim
        out = self.l1(z)
        if self.verbose: print("l1 out : ",out.shape)
        out = out.view(out.shape[0], self.max_filters, self.init_size, self.init_size)
        # Dim : (self.max_filters, opt.img_size/8, opt.img_size/8)
        if self.verbose: print("View out : ",out.shape)

        out = self.conv1(out)
        # Dim : (self.max_filters/2, opt.img_size/4, opt.img_size/4)
        if self.verbose: print("Conv1 out : ",out.shape)
        out = self.conv2(out)
        # Dim : (self.max_filters/4, opt.img_size/2, opt.img_size/2)
        if self.verbose: print("Conv2 out : ",out.shape)
        out = self.conv3(out)
        # Dim : (self.max_filters/8, opt.img_size, opt.img_size)
        if self.verbose: print("Conv3 out : ",out.shape)

        img = self.conv_blocks(out)
        # Dim : (opt.chanels, opt.img_size, opt.img_size)
        if self.verbose: print("img out : ", img.shape)

        return img

    def _name(self):
        return "Generator"

class Discriminator(nn.Module):
    def __init__(self,verbose=verbose):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, **opts_conv), NL]#, nn.Dropout2d(0.25)
            if bn:
                block.append(nn.BatchNorm2d(out_filters, opt.eps))
            return block

        self.max_filters = 512
        self.verbose = verbose

        self.conv1 = nn.Sequential(*discriminator_block(opt.channels, 64, bn=False),)
        self.conv2 = nn.Sequential(*discriminator_block(64, 128),)
        self.conv3 = nn.Sequential(*discriminator_block(128, 256),)
        self.conv4 = nn.Sequential(*discriminator_block(256, self.max_filters),)

        # The height and width of downsampled image
        self.init_size = opt.img_size // opts_conv['stride']**4
        self.adv_layer = nn.Sequential(nn.Linear(self.max_filters * self.init_size ** 2, 1))#, nn.Sigmoid()

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
            # Dim : (self.max_filters/8, opt.img_size/2, opt.img_size/2)
            out = self.conv2(out)
            # Dim : (self.max_filters/4, opt.img_size/4, opt.img_size/4)
            out = self.conv3(out)
            # Dim : (self.max_filters/2, opt.img_size/4, opt.img_size/4)
            out = self.conv4(out)
            # Dim : (self.max_filters, opt.img_size/8, opt.img_size/8)

            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)
            # Dim : (1)

        return validity

    def _name(self):
        return "Discriminator"

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()
MSE_loss = torch.nn.MSELoss()
sigmoid = nn.Sigmoid()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
encoder = Encoder()

print_network(generator)
print_network(discriminator)
print_network(encoder)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    encoder.cuda()
    MSE_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
encoder.apply(weights_init_normal)

# Configure data loader
dataloader = load_data(depth + "../../cropped_clear/cp/", opt.img_size, opt.batch_size, rand_hflip=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lrG, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lrD, betas=(opt.b1, opt.b2))
optimizer_E = torch.optim.Adam(itertools.chain(encoder.parameters(), generator.parameters()), lr=opt.lrE, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

start_epoch = 1

# ----------
#  Training
# ----------
nb_batch = len(dataloader)
nb_epochs = 1 + opt.n_epochs - start_epoch

hist = init_hist(nb_epochs, nb_batch, lossE=True)

save_dot = 1 # Nombre d'epochs avant de sauvegarder un point des courbes
batch_on_save_dot = save_dot*len(dataloader)

# Vecteur z fixe pour faire les samples
N_samples = 5**2
fixed_noise = Variable(Tensor(np.random.normal(0, 1, (N_samples, opt.latent_dim))))

t_total = time.time()
for j, epoch in enumerate(range(start_epoch, opt.n_epochs + 1)):
    t_epoch = time.time()
    for i, (imgs, _) in enumerate(dataloader):
        t_batch = time.time()
        # ---------------------
        #  Train Encoder
        # ---------------------

        real_imgs = Variable(imgs.type(Tensor))

        optimizer_E.zero_grad()
        z_imgs = encoder(real_imgs)
        decoded_imgs = generator(z_imgs)

        # Loss measures Encoder's ability to generate vectors suitable with the generator
        # TODO add a loss for the distribution of z values
        z_zeros = Variable(Tensor(z_imgs.size(0), z_imgs.size(1)).fill_(0), requires_grad=False)
        z_ones = Variable(Tensor(z_imgs.size(0), z_imgs.size(1)).fill_(1), requires_grad=False)
        e_loss = MSE_loss(real_imgs, decoded_imgs) + MSE_loss(z_imgs, z_zeros) + MSE_loss(z_imgs.pow(2), z_ones).pow(.5)
        # print("e_loss out : ", e_loss.shape)
        # e_lambda_norm = torch.tensor(1.)
        # e_lambda_dev = torch.tensor(1.)
        # std = torch.sqrt(z_imgs.pow(2).mean(1))
        # print("std out : ", std.shape)
        # print("std.pow(2) out : ", std.pow(2).shape)
        # e_loss += e_lambda_dev * std.pow(2)
        # e_loss += e_lambda_norm * (std - 1).pow(2)

        # import numpy as np
        # import torch
        # Tensor =  torch.FloatTensor
        # from torch.autograd import Variable
        # z = Variable(Tensor(z))
        # loss = nn.MSELoss()
        # import torch.nn as nn
        # loss = nn.MSELoss()
        # z_zeros = Variable(Tensor(400, 100).fill_(0), requires_grad=False)
        # z = np.random.normal(0, 1, (400, 100))
        # z.mean()
        # z.std()
        # z = Variable(Tensor(z))
        # loss(z, z_zeros)
        # z_ones = Variable(Tensor(400, 100).fill_(1), requires_grad=False)
        # loss(z.pow(2), z_ones)
        # # Backward
        # e_loss.backward()

        optimizer_E.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Adversarial ground truths
        valid_smooth = Variable(Tensor(imgs.shape[0], 1).fill_(float(np.random.uniform(0.9, 1.0, 1))), requires_grad=False)
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        # Generate a batch of images
        if False:
            z = np.random.uniform(0, 1, (imgs.shape[0], opt.latent_dim))
        else:
            z = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
        z = Variable(Tensor(z))
        gen_imgs = generator(z)

        optimizer_D.zero_grad()

        # Real batch
        #Discriminator descision
        d_x = discriminator(real_imgs)
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(d_x, valid_smooth)
        # Backward
        real_loss.backward()

        # Fake batch
        #Discriminator descision
        d_g_z = discriminator(gen_imgs.detach())
        # Measure discriminator's ability to classify real from generated samples
        fake_loss = adversarial_loss(d_g_z, fake)
        # Backward
        fake_loss.backward()

        d_loss = real_loss + fake_loss

        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # New discriminator descision, Since we just updated D
        d_g_z = discriminator(gen_imgs)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(d_g_z, valid)
        # Backward
        g_loss.backward()

        optimizer_G.step()


        print(
            "[Epoch %d/%d] [Batch %d/%d] [E loss: %f] [D loss: %f] [G loss: %f] [Time: %fs]"
            % (epoch, opt.n_epochs, i+1, len(dataloader), e_loss.item(), d_loss.item(), g_loss.item(), time.time()-t_batch)
        )

        # Compensation pour le BCElogits
        d_x = sigmoid(d_x)
        d_g_z = sigmoid(d_g_z)

        # Save Losses and scores for plotting later
        save_hist_batch(hist, i, j, g_loss, d_loss, d_x, d_g_z, e_loss)

    # Save Losses and scores for plotting later
    save_hist_epoch(hist, j, E_losses=True)

    # Save samples
    if epoch % opt.sample_interval == 0:
        sampling(fixed_noise, generator, opt.sample_path, epoch, tag)

    # Intermediate plot
    if epoch % (opt.n_epochs/4) == 0:
        do_plot(hist, start_epoch, epoch, E_losses=True)

    print("[Epoch Time: ",time.time()-t_epoch,"s]")

print("[Total Time: ",time.strftime("%Hh:%Mm:%Ss",time.gmtime(time.time()-t_total)),"]")

#Plot
do_plot(hist, start_epoch, epoch, E_losses=True)

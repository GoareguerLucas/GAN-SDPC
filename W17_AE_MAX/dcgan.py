import argparse
import os
import numpy as np

import itertools

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
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--runs_path", type=str, default='AE_MAX/200e16i128b/',
                    help="Dossier de stockage des résultats sous la forme : Experience_names/parameters/")
parser.add_argument("-e", "--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("-b", "--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lrD", type=float, default=0.00005, help="adam: learning rate for D")
parser.add_argument("--lrG", type=float, default=0.00015, help="adam: learning rate for G")
parser.add_argument("--lrE", type=float, default=0.00015, help="adam: learning rate for E")
parser.add_argument("--eps", type=float, default=0.0, help="batchnorm: espilon for numerical stability")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lrelu", type=float, default=0.01, help="LeakyReLU : alpha")
parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent space")
parser.add_argument("-i", "--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("-s", "--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--sample_path", type=str, default='images')
parser.add_argument("-m", "--model_save_interval", type=int, default=150,
                    help="interval between image sampling. If model_save_interval > n_epochs : no save")
parser.add_argument('--model_save_path', type=str, default='models')
parser.add_argument('--load_model', action="store_true",
                    help="Load model present in model_save_path/Last_*.pt, if present.")
parser.add_argument("-d", "--depth", action="store_true",
                    help="Utiliser si utils.py et SimpsonsDataset.py sont deux dossier au dessus.")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Afficher des informations complémentaire.")
parser.add_argument("--GPU", type=int, default=0, help="Identifiant du GPU à utiliser.")
opt = parser.parse_args()
print(opt)

# Particular import
depth = ""
if opt.depth == True:
    depth = "../"
sys.path.append(depth + "../")  # ../../GAN-SDPC/

from SimpsonsDataset import SimpsonsDataset, FastSimpsonsDataset
from utils import *
from plot import *

# Dossier de sauvegarde
os.makedirs(opt.model_save_path, exist_ok=True)

# Gestion du time tag
try:
    tag = datetime.datetime.now().isoformat(sep='_', timespec='seconds')
except TypeError:
    # Python 3.5 and below
    # 'timespec' is an invalid keyword argument for this function
    tag = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_')
tag = tag.replace(':','.')



cuda = True if torch.cuda.is_available() else False
NL = nn.LeakyReLU(opt.lrelu, inplace=True)
opts_conv = dict(kernel_size=9, stride=2, padding=4, padding_mode='zeros')
channels = [64, 128, 256, 512]

class Encoder(nn.Module):
    def __init__(self, verbose=opt.verbose):
        super(Encoder, self).__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, **opts_conv), NL]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, opt.eps))
            return block

        self.verbose = verbose
        # use a different layer in the encoder using similarly max_filters
        # channels[3] = 512

        self.conv1 = nn.Sequential(*encoder_block(opt.channels, channels[0], bn=False),)
        self.conv2 = nn.Sequential(*encoder_block(channels[0], channels[1]),)
        self.conv3 = nn.Sequential(*encoder_block(channels[1], channels[2]),)
        self.conv4 = nn.Sequential(*encoder_block(channels[2], channels[3]),)

        self.init_size = opt.img_size // opts_conv['stride']**4
        self.vector = nn.Linear(channels[3] * self.init_size ** 2, opt.latent_dim)
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
        if self.verbose: print("Z : ",z.shape)

        return z

class Generator(nn.Module):
    def __init__(self, verbose=opt.verbose):
        super(Generator, self).__init__()

        # def generator_block(in_filters, out_filters):
        #     block = [nn.ConvTranspose2d(in_filters, out_filters, **opts_conv, output_padding=1), nn.BatchNorm2d(out_filters, opt.eps), NL]
        #
        #     return block
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
    #print("Nombre de GPU : ",torch.cuda.device_count())
    if torch.cuda.device_count() > opt.GPU:
        torch.cuda.set_device(opt.GPU)

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

# ----------
#  Load models
# ----------
start_epoch = 1
if opt.load_model == True:
    start_epoch = load_models(discriminator, optimizer_D, generator, optimizer_G, opt.n_epochs, opt.model_save_path)

# ----------
#  Tensorboard
# ----------
path_data1 = depth + "../runs/" + opt.runs_path
path_data2 = depth + "../runs/" + opt.runs_path + tag[:-1] + "/"

# Les runs sont sauvegarder dans un dossiers "runs" à la racine du projet, dans un sous dossiers opt.runs_path.
os.makedirs(path_data1, exist_ok=True)
os.makedirs(path_data2, exist_ok=True)

writer = SummaryWriter(log_dir=path_data2)

# ----------
#  Training
# ----------

nb_batch = len(dataloader)
nb_epochs = 1 + opt.n_epochs - start_epoch

hist = init_hist(nb_epochs, nb_batch, lossE=True)

save_dot = 1 # Nombre d'epochs avant de sauvegarder un point des courbes
batch_on_save_dot = save_dot*len(dataloader)

# Vecteur z fixe pour faire les samples
N_samples = 24
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
        # DONE add a loss for the distance between of z values
        z_zeros = Variable(Tensor(z_imgs.size(0), z_imgs.size(1)).fill_(0), requires_grad=False)
        z_ones = Variable(Tensor(z_imgs.size(0), z_imgs.size(1)).fill_(1), requires_grad=False)
        e_loss = MSE_loss(real_imgs, decoded_imgs) + MSE_loss(z_imgs, z_zeros) + MSE_loss(z_imgs.pow(2), z_ones).pow(.5)

        # Backward
        e_loss.backward()

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
        z = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
        z = Variable(Tensor(z))
        gen_imgs = generator(z)

        optimizer_D.zero_grad()

        # Real batch
        # Discriminator descision
        d_x = discriminator(real_imgs)
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(d_x, valid_smooth)
        # Backward
        real_loss.backward()

        # Fake batch
        # Discriminator descision
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

        # Save Losses and scores for Tensorboard
        save_hist_batch(hist, i, j, g_loss, d_loss, d_x, d_g_z)

        # Tensorboard save
        iteration = i + nb_batch * j
        writer.add_scalar('e_loss', e_loss.item(), global_step=iteration)
        writer.add_scalar('g_loss', g_loss.item(), global_step=iteration)
        writer.add_scalar('d_loss', d_loss.item(), global_step=iteration)

        writer.add_scalar('d_x_mean', hist["d_x_mean"][i], global_step=iteration)
        writer.add_scalar('d_g_z_mean', hist["d_g_z_mean"][i], global_step=iteration)

        writer.add_scalar('d_x_cv', hist["d_x_cv"][i], global_step=iteration)
        writer.add_scalar('d_g_z_cv', hist["d_g_z_cv"][i], global_step=iteration)

        writer.add_histogram('D(x)', d_x, global_step=iteration)
        writer.add_histogram('D(G(z))', d_g_z, global_step=iteration)

    writer.add_scalar('D_x_max', hist["D_x_max"][j], global_step=epoch)
    writer.add_scalar('D_x_min', hist["D_x_min"][j], global_step=epoch)
    writer.add_scalar('D_G_z_min', hist["D_G_z_min"][j], global_step=epoch)
    writer.add_scalar('D_G_z_max', hist["D_G_z_max"][j], global_step=epoch)

    # Save samples
    if epoch % opt.sample_interval == 0:
        tensorboard_sampling(fixed_noise, generator, writer, epoch)

    # Save models
    if epoch % opt.model_save_interval == 0:
        num = str(int(epoch / opt.model_save_interval))
        save_model(discriminator, optimizer_D, epoch, opt.model_save_path + "/" + num + "_D.pt")
        save_model(generator, optimizer_G, epoch, opt.model_save_path + "/" + num + "_G.pt")
        save_model(encoder, optimizer_E, epoch, opt.model_save_path + "/" + num + "_E.pt")

    print("[Epoch Time: ", time.time() - t_epoch, "s]")

durer = time.gmtime(time.time() - t_total)
print("[Total Time: ", durer.tm_mday - 1, "j:", time.strftime("%Hh:%Mm:%Ss", durer), "]", sep='')

# Save model for futur training
if opt.model_save_interval < opt.n_epochs + 1:
    save_model(discriminator, optimizer_D, epoch, opt.model_save_path + "/last_D.pt")
    save_model(generator, optimizer_G, epoch, opt.model_save_path + "/last_G.pt")
    save_model(encoder, optimizer_E, epoch, opt.model_save_path + "/last_E.pt")

writer.close()

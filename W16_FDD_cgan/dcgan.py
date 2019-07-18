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
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--runs_path", type=str, default='CGAN/200e64i64b/',
                    help="Dossier de stockage des résultats sous la forme : Experience_names/parameters/")
parser.add_argument("-e", "--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lrD", type=float, default=0.00004, help="adam: learning rate for D")
parser.add_argument("--lrG", type=float, default=0.0004, help="adam: learning rate for G")
parser.add_argument("--eps", type=float, default=0.1, help="batchnorm: espilon for numerical stability")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lrelu", type=float, default=0.2, help="LeakyReLU : alpha")
parser.add_argument("--latent_dim", type=int, default=6, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=3, help="number of classes for dataset")
parser.add_argument("-i", "--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("-s", "--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--sample_path", type=str, default='images')
parser.add_argument("-m", "--model_save_interval", type=int, default=2500,
                    help="interval between image sampling. If model_save_interval > n_epochs, no save")
parser.add_argument('--model_save_path', type=str, default='models')
parser.add_argument('--load_model', action="store_true",
                    help="Load model present in model_save_path/Last_*.pt, if present.")
parser.add_argument("-d", "--depth", action="store_true",
                    help="Utiliser si utils.py et SimpsonsDataset.py sont deux dossier au dessus.")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Afficher des informations complémentaire.")
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
# (N + 2*p - k) / s +1 cf https://pytorch.org/docs/stable/nn.html#conv2d
opts_conv = dict(kernel_size=5, stride=2, padding=2, padding_mode='zeros')
# verbose=True
channels = [16, 32, 64, 128]
channels = [64, 128, 256, 512]

class Generator(nn.Module):
    def __init__(self, verbose=opt.verbose):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        
        def generator_block(in_filters, out_filters):
            block = [nn.UpsamplingNearest2d(scale_factor=opts_conv['stride']), nn.Conv2d(in_filters, out_filters, kernel_size=opts_conv['kernel_size'], stride=1, padding=opts_conv['padding'], padding_mode=opts_conv['padding_mode']), nn.BatchNorm2d(out_filters, opt.eps), NL]

            return block

        self.verbose = verbose
        self.init_size = opt.img_size // opts_conv['stride']**3
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim+opt.n_classes, channels[3] * self.init_size ** 2), NL)


        self.conv1 = nn.Sequential(*generator_block(channels[3], channels[2]),)
        self.conv2 = nn.Sequential(*generator_block(channels[2], channels[1]),)
        self.conv3 = nn.Sequential(*generator_block(channels[1], channels[0]),)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(channels[0], opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, z, labels):
        if self.verbose: print("G")
        if self.verbose: print("z and labels : ", z.shape, labels.shape)
        gen_input = torch.cat((z, self.label_emb(labels)), -1)
        if self.verbose: print("gen_input out : ",gen_input.shape)
        # Dim : opt.latent_dim
        out = self.l1(gen_input)
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
        
        self.data = discriminator_block(opt.channels, channels[0], bn=False)
        self.label = discriminator_block(opt.channels, channels[0], bn=False)
        
        self.conv2 = nn.Sequential(*discriminator_block(channels[0], channels[1]),)
        self.conv3 = nn.Sequential(*discriminator_block(channels[1], channels[2]),)
        self.conv4 = nn.Sequential(*discriminator_block(channels[2], channels[3]),)

        # The height and width of downsampled image
        self.init_size = opt.img_size // opts_conv['stride']**4
        self.adv_layer = nn.Sequential(nn.Linear(channels[3] * self.init_size ** 2, 1))#, nn.Sigmoid()

    def forward(self, img, labels):
        if self.verbose:
            print("D")
            print("Image shape : ",img.shape)
            
            x = self.data(img)
            y = self.label(labels)
            print("Conv1 data : ",x.shape)
            print("Conv1 label : ",y.shape)
            
            # concat conv1_data and Conv1_label
            out = torch.cat((x, y), dim=1)
            print("Cat : ",out.shape)
            
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
sigmoid = nn.Sigmoid()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

print_network(generator)
print_network(discriminator)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = load_data(depth + "../../FDD/inferno/", opt.img_size, opt.batch_size, rand_hflip=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lrG, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lrD, betas=(opt.b1, opt.b2))

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

LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

nb_batch = len(dataloader)
nb_epochs = 1 + opt.n_epochs - start_epoch

hist = init_hist(nb_epochs, nb_batch)

# Vecteur z fixe pour faire les samples
fixed_noise = Variable(Tensor(np.random.normal(0, 1, (24, opt.latent_dim))))

t_total = time.time()
for j, epoch in enumerate(range(start_epoch, opt.n_epochs + 1)):
    t_epoch = time.time()
    for i, (imgs, _) in enumerate(dataloader):
        t_batch = time.time()
        
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, opt.batch_size)))
        
        # Adversarial ground truths
        valid_smooth = Variable(Tensor(imgs.shape[0], 1).fill_(
            float(np.random.uniform(0.9, 1.0, 1))), requires_grad=False)
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        # Generate a batch of images
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_imgs = generator(z, gen_labels)
        
        #print("Max : ",gen_imgs.max()," Min :",gen_imgs.min())
        
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real batch
        # Discriminator descision
        d_x = discriminator(real_imgs, gen_labels)
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(d_x, valid_smooth)
        # Backward
        real_loss.backward()

        # Fake batch
        # Discriminator descision
        d_g_z = discriminator(gen_imgs.detach(), gen_labels)
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
        d_g_z = discriminator(gen_imgs, gen_labels)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(d_g_z, valid)
        # Backward
        g_loss.backward()

        optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time: %fs]"
            % (epoch, opt.n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item(), time.time() - t_batch)
        )

        # Compensation pour le BCElogits
        d_x = sigmoid(d_x)
        d_g_z = sigmoid(d_g_z)

        # Save Losses and scores for Tensorboard
        save_hist_batch(hist, i, j, g_loss, d_loss, d_x, d_g_z)

        # Tensorboard save
        iteration = i + nb_batch * j
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

    print("[Epoch Time: ", time.time() - t_epoch, "s]")

durer = time.gmtime(time.time() - t_total)
print("[Total Time: ", durer.tm_mday - 1, "j:", time.strftime("%Hh:%Mm:%Ss", durer), "]", sep='')

# Save model for futur training
if opt.model_save_interval < opt.n_epochs + 1:
    save_model(discriminator, optimizer_D, epoch, opt.model_save_path + "/last_D.pt")
    save_model(generator, optimizer_G, epoch, opt.model_save_path + "/last_G.pt")

writer.close()

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
timetag = datetime.datetime.now().isoformat(timespec='seconds') + '_'

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--runs_path", type=str, default='current/200e64i64b/', help="Dossier de stockage des résultats sous la forme : Experience_names/parameters/")
parser.add_argument("-e", "--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lrD", type=float, default=0.00004, help="adam: learning rate for D")
parser.add_argument("--lrG", type=float, default=0.0004, help="adam: learning rate for G")
parser.add_argument("--eps", type=float, default=0.00005, help="batchnorm: espilon for numerical stability")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("-i", "--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("-s", "--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--sample_path", type=str, default='images')
parser.add_argument("-m", "--model_save_interval", type=int, default=2500, help="interval between image sampling. If model_save_interval > n_epochs, no save")
parser.add_argument('--model_save_path', type=str, default='models')
parser.add_argument('--load_model', action="store_true",
                    help="Load model present in model_save_path/Last_*.pt, if present.")
parser.add_argument("-d", "--depth", action="store_true",
                    help="Utiliser si utils.py et SimpsonsDataset.py sont deux dossier au dessus.")
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
os.makedirs(opt.sample_path, exist_ok=True)
os.makedirs(opt.model_save_path, exist_ok=True)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m, factor=1.0):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        n = float(m.in_channels * m.kernel_size[0] * m.kernel_size[1])
        n += float(m.kernel_size[0] * m.kernel_size[1] * m.out_channels)
        n = n / 2.0
        m.weight.data.normal_(0, np.sqrt(factor / n))
        m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        n = float(m.in_features + m.out_features)
        n = n / 2.0
        m.weight.data.normal_(0, np.sqrt(factor / n))
        m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, verbose=False):
        super(Generator, self).__init__()

        def generator_block(in_filters, out_filters, kernel=4, stride=2):
            #block = [nn.Conv2d(in_filters, out_filters, kernel, stride=stride, padding=2), nn.Upsample(scale_factor=2), nn.BatchNorm2d(out_filters, opt.eps), nn.LeakyReLU(0.2, inplace=True)]
            block = [nn.ConvTranspose2d(in_filters, out_filters, kernel, stride=stride, padding=1),
                     nn.BatchNorm2d(out_filters, opt.eps), nn.LeakyReLU(0.2, inplace=True)]

            return block

        self.verbose = verbose

        self.max_filters = 512
        self.init_size = opt.img_size // 8
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.max_filters *
                                          self.init_size ** 2), nn.LeakyReLU(0.2, inplace=True))

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
		if self.verbose: print("l1 out : ", out.shape)
		out = out.view(out.shape[0], self.max_filters, self.init_size, self.init_size)
		# Dim : (self.max_filters, opt.img_size/8, opt.img_size/8)
		
		if self.verbose: print("View out : ", out.shape)
		out = self.conv1(out)
		# Dim : (self.max_filters/2, opt.img_size/4, opt.img_size/4)
		if self.verbose: print("Conv1 out : ", out.shape)
		out = self.conv2(out)
		# Dim : (self.max_filters/4, opt.img_size/2, opt.img_size/2)
		if self.verbose: print("Conv2 out : ", out.shape)
		out = self.conv3(out)
		# Dim : (self.max_filters/8, opt.img_size, opt.img_size)
		if self.verbose: print("Conv3 out : ", out.shape)

		img = self.conv_blocks(out)
		# Dim : (opt.chanels, opt.img_size, opt.img_size)
		if self.verbose: print("Channels Conv out : ", img.shape)

    def _name(self):
        return "Generator"


class Discriminator(nn.Module):
    def __init__(self, verbose=False):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True, kernel=4, stride=2, padding=1):
            block = [nn.Conv2d(in_filters, out_filters, kernel, stride, padding=padding),
                     nn.LeakyReLU(0.2, inplace=True)]  # , nn.Dropout2d(0.25)
            if bn:
                block.append(nn.BatchNorm2d(out_filters, opt.eps))
            return block

        self.max_filters = 512
        self.verbose = verbose

        self.conv1 = nn.Sequential(*discriminator_block(opt.channels, 64, bn=False),)
        self.conv2 = nn.Sequential(*discriminator_block(64, 128),)
        self.conv3 = nn.Sequential(*discriminator_block(128, 256, stride=1, padding=2),)
        self.conv4 = nn.Sequential(*discriminator_block(256, self.max_filters),)

        # The height and width of downsampled image
        self.init_size = opt.img_size // 8
        self.adv_layer = nn.Sequential(nn.Linear(self.max_filters * self.init_size ** 2, 1))  # , nn.Sigmoid()

    def forward(self, img):
		if self.verbose: print("D")
		# Dim : (opt.chanels, opt.img_size, opt.img_size)
		if self.verbose: print("Image shape : ", img.shape)
		
		out = self.conv1(img)
		# Dim : (self.max_filters/8, opt.img_size/2, opt.img_size/2)
		if self.verbose: print("Conv1 out : ", out.shape)
		out = self.conv2(out)
		# Dim : (self.max_filters/4, opt.img_size/4, opt.img_size/4)
		if self.verbose: print("Conv2 out : ", out.shape)
		out = self.conv3(out)
		# Dim : (self.max_filters/2, opt.img_size/4, opt.img_size/4)
		if self.verbose: print("Conv3 out : ", out.shape)
		out = self.conv4(out)
		# Dim : (self.max_filters, opt.img_size/8, opt.img_size/8)
		if self.verbose: print("Conv4 out : ", out.shape)

		out = out.view(out.shape[0], -1)
		if self.verbose: print("View out : ", out.shape)
		validity = self.adv_layer(out)
		# Dim : (1)
		if self.verbose: print("Val out : ", validity.shape)

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
dataloader = load_data(depth + "../../cropped/cp/", opt.img_size, opt.batch_size, rand_hflip=True)

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
# Les runs sont sauvegarder dans un dossiers "runs" à la racine du projet, dans un sous dossiers opt.runs_path. 
os.makedirs(depth+"../runs/"+opt.runs_path, exist_ok=True)

writer = SummaryWriter(log_dir=depth+"../runs/"+opt.runs_path)

# ----------
#  Training
# ----------

nb_batch = len(dataloader)
nb_epochs = 1 + opt.n_epochs - start_epoch

hist = init_hist(nb_epochs, nb_batch)

# Vecteur z fixe pour faire les samples
fixed_noise = Variable(Tensor(np.random.normal(0, 1, (25, opt.latent_dim))))

t_total = time.time()
for j, epoch in enumerate(range(start_epoch, opt.n_epochs + 1)):
    t_epoch = time.time()
    for i, (imgs, _) in enumerate(dataloader):
        t_batch = time.time()

        # Adversarial ground truths
        valid_smooth = Variable(Tensor(imgs.shape[0], 1).fill_(
            float(np.random.uniform(0.9, 1.0, 1))), requires_grad=False)
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        # Generate a batch of images
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_imgs = generator(z)

        # ---------------------
        #  Train Discriminator
        # ---------------------

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
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time: %fs]"
            % (epoch, opt.n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item(), time.time() - t_batch)
        )

        # Compensation pour le BCElogits
        d_x = sigmoid(d_x)
        d_g_z = sigmoid(d_g_z)

        # Save Losses and scores for plotting later
        save_hist_batch(hist, i, j, g_loss, d_loss, d_x, d_g_z)

        # Tensorboard test
        iteration = i + nb_batch*j
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

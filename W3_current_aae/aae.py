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
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lrD", type=float, default=0.0004, help="adam: learning rate for D")
parser.add_argument("--lrG", type=float, default=0.0004, help="adam: learning rate for G")
parser.add_argument("--eps", type=float, default=0.00005, help="batchnorm: espilon for numerical stability")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=int, default=1, help="number of gpu use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
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

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def reparameterization(mu, logvar):
	std = torch.exp(logvar / 2)
	sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
	z = sampled_z * std + mu
	return z


class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(int(np.prod(img_shape)), 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512, opt.eps),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.mu = nn.Linear(512, opt.latent_dim)
		self.logvar = nn.Linear(512, opt.latent_dim)

	def forward(self, img):
		img_flat = img.view(img.shape[0], -1)
		
		if img_flat.is_cuda and opt.n_gpu > 1:
			x = nn.parallel.data_parallel(self.model, img_flat, range(opt.n_gpu))
		else:
			x = self.model(img_flat)
			
		mu = self.mu(x)
		logvar = self.logvar(x)	
		z = reparameterization(mu, logvar)
		return z


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(opt.latent_dim, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512, opt.eps),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, int(np.prod(img_shape))),
			nn.Tanh(),
		)

	def forward(self, z):
		if z.is_cuda and opt.n_gpu > 1:
			img_flat = nn.parallel.data_parallel(self.model, z, range(opt.n_gpu))
		else:
			img_flat = self.model(z)
			
		img = img_flat.view(img_flat.shape[0], *img_shape)
		return img


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(opt.latent_dim, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid(),
		)

	def forward(self, z):
		if z.is_cuda and opt.n_gpu > 1:
			validity = nn.parallel.data_parallel(self.model, z, range(opt.n_gpu))
		else:
			validity = self.model(z)
		
		#validity = self.model(z)
		return validity


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
	encoder.cuda()
	decoder.cuda()
	discriminator.cuda()
	adversarial_loss.cuda()
	pixelwise_loss.cuda()

# Configure data loader
dataloader = load_data("../../cropped/cp/",opt.img_size,opt.batch_size)

# Optimizers
optimizer_G = torch.optim.Adam(
	itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lrG, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lrD, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_image(n_row, epoch_done):
	"""Saves a grid of generated digits"""
	# Sample noise
	z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
	gen_imgs = decoder(z)
	save_image(gen_imgs.data, "%s/%d.png" % (opt.sample_path, epoch_done), nrow=n_row, normalize=True)


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
	for i, (imgs, _) in enumerate(dataloader):
		t_batch = time.time()
		
		# Adversarial ground truths
		valid_smooth = Variable(Tensor(imgs.shape[0], 1).fill_(float(np.random.uniform(0.7, 1.0, 1))), requires_grad=False)
		valid = Variable(Tensor(imgs.size(0), 1).fill_(1), requires_grad=False)
		fake = Variable(Tensor(imgs.size(0), 1).fill_(0), requires_grad=False)

		# Configure input
		real_imgs = Variable(imgs.type(Tensor))
		
		# -----------------
		#  Train Generator
		# -----------------

		optimizer_G.zero_grad()

		encoded_imgs = encoder(real_imgs)
		decoded_imgs = decoder(encoded_imgs)

		# Loss measures generator's ability to fool the discriminator
		g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
			decoded_imgs, real_imgs
		)

		g_loss.backward()
		optimizer_G.step()
		
		# ---------------------
		#  Train Discriminator
		# ---------------------

		optimizer_D.zero_grad()

		# Sample noise as discriminator ground truth
		z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
		
		#Discriminator descision
		d_x_tmp = discriminator(z)
		d_g_x_tmp = discriminator(encoded_imgs.detach())
		
		# Measure discriminator's ability to classify real from generated samples
		real_loss = adversarial_loss(d_x_tmp, valid_smooth)
		fake_loss = adversarial_loss(d_g_x_tmp, fake)
		d_loss = 0.5 * (real_loss + fake_loss)

		d_loss.backward()
		optimizer_D.step()
		
		print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time: %fs]"
			% (epoch, opt.n_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item(), time.time()-t_batch)
		)
		
		# Save samples
		if epoch % opt.sample_interval == 0 and i == 0:
			sample_image(n_row=5, epoch=epoch)
		
		# Save Losses and scores for plotting later
		g_losses.append(g_loss.item())
		d_losses.append(d_loss.item())
		d_x.append(torch.sum(discriminator(z)).item()/imgs.size(0))
		d_g_z.append(torch.sum(discriminator(encoded_imgs.detach())).item()/imgs.size(0))
		if epoch % save_dot == 0 and i == 0:
			G_losses.append(sum(g_losses)/batch_on_save_dot)
			D_losses.append(sum(d_losses)/batch_on_save_dot)
			g_losses = []
			d_losses = []
			D_x.append(sum(d_x)/batch_on_save_dot)
			D_G_z.append(sum(d_g_z)/batch_on_save_dot)
			d_x = []
			d_g_z = []
			
		# Save models
		if epoch % opt.model_save_interval == 0 and i == 0:
			num = str(int(epoch / opt.model_save_interval))
			save_model(discriminator,optimizer_D,epoch,opt.model_save_path+"/"+num+"_D.pt")
			save_model(encoder,optimizer_G,epoch,opt.model_save_path+"/"+num+"_encoder.pt")
			save_model(decoder,optimizer_G,epoch,opt.model_save_path+"/"+num+"_decoder.pt")
		
		# Intermediate plot
		if epoch % (opt.n_epochs/4) == 0 and i == 0:
			#Plot losses			
			plot_losses(G_losses,D_losses)
			#Plot scores
			plot_scores(D_x,D_G_z)
		
	print("[Epoch Time: ",time.time()-t_epoch,"s]")

print("[Total Time: ",time.time()-t_total,"s]")

#Plot losses			
plot_losses(G_losses,D_losses)

#Plot game score
plot_scores(D_x,D_G_z)

save_model(discriminator,optimizer_D,epoch,opt.model_save_path+"/last_D.pt")
save_model(encoder,optimizer_G,epoch,opt.model_save_path+"/last_encoder.pt")
save_model(decoder,optimizer_G,epoch,opt.model_save_path+"/last_decoder.pt")

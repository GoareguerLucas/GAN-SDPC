import argparse
import os
import numpy as np


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
parser.add_argument("--eps", type=float, default=0.00005, help="batchnorm: espilon for numerical stability")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
parser.add_argument("--sample_path", type=str, default='images')
parser.add_argument("--model_save_interval", type=int, default=2500, help="interval between image sampling")
parser.add_argument('--model_save_path', type=str, default='models')
opt = parser.parse_args()
print(opt)

# Dossier de sauvegarde
os.makedirs(opt.sample_path, exist_ok=True)
os.makedirs(opt.model_save_path, exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		self.init_size = opt.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks = nn.Sequential(
			nn.BatchNorm2d(128),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
			nn.Tanh(),
		)

	def forward(self, noise):
		out = self.l1(noise)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks(out)
		return img


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		# Upsampling
		self.down = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 2, 1), nn.ReLU())
		# Fully-connected layers
		self.down_size = opt.img_size // 2
		down_dim = 64 * (opt.img_size // 2) ** 2
		self.fc = nn.Sequential(
			nn.Linear(down_dim, 32),
			nn.BatchNorm1d(32, 0.8),
			nn.ReLU(inplace=True),
			nn.Linear(32, down_dim),
			nn.BatchNorm1d(down_dim),
			nn.ReLU(inplace=True),
		)
		# Upsampling
		self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, opt.channels, 3, 1, 1))

	def forward(self, img):
		out = self.down(img)
		out = self.fc(out.view(out.size(0), -1))
		out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
		return out


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
	generator.cuda()
	discriminator.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = load_data("../../cropped/cp/",opt.img_size,opt.batch_size)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lrG, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lrD, betas=(opt.b1, opt.b2))

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

# BEGAN hyper parameters
gamma = 0.9
lambda_k = 0.001
k = 0.0

k_tmp = []
M_tmp = []
k_plot = []
M_plot = []

t_total = time.time()
for epoch in range(1,opt.n_epochs+1):
	t_epoch = time.time()
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
		gen_imgs = generator(z)

		# Loss measures generator's ability to fool the discriminator
		g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

		g_loss.backward()
		optimizer_G.step()

		# ---------------------
		#  Train Discriminator
		# ---------------------

		optimizer_D.zero_grad()

		# Measure discriminator's ability to classify real from generated samples
		d_real = discriminator(real_imgs)
		d_fake = discriminator(gen_imgs.detach())

		d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
		d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
		d_loss = d_loss_real - k * d_loss_fake

		d_loss.backward()
		optimizer_D.step()

		# ----------------
		# Update weights
		# ----------------

		diff = torch.mean(gamma * d_loss_real - d_loss_fake)

		# Update weight term for fake samples
		k = k + lambda_k * diff.item()
		k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

		# Update convergence metric
		M = (d_loss_real + torch.abs(diff)).item()

		# --------------
		# Log Progress
		# --------------

		print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time: %fs] -- M: %f, k: %f"
			% (epoch, opt.n_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item(), time.time()-t_batch, M, k)
		)
	
		# Save Losses and scores for plotting later
		M_tmp.append(M)
		k_tmp.append(k)
		g_losses.append(g_loss.item())
		d_losses.append(d_loss.item())
		d_x.append(torch.sum(d_real).item()/imgs.size(0))
		d_g_z.append(torch.sum(d_fake).item()/imgs.size(0))
		
	# Save samples
	if epoch % opt.sample_interval == 0:
		save_image(gen_imgs.data[:25], "%s/%d.png" % (opt.sample_path, epoch), nrow=5, normalize=True)
	
	# Save Losses and scores for plotting later
	if epoch % save_dot == 0:
		G_losses.append(sum(g_losses)/batch_on_save_dot)
		D_losses.append(sum(d_losses)/batch_on_save_dot)
		g_losses = []
		d_losses = []
		D_x.append(sum(d_x)/batch_on_save_dot)
		D_G_z.append(sum(d_g_z)/batch_on_save_dot)
		d_x = []
		d_g_z = []
		
		M_plot.append(sum(M_tmp)/batch_on_save_dot)
		k_plot.append(sum(k_tmp)/batch_on_save_dot)
		k_tmp = []
		M_tmp = []

	
	# Save models
	if epoch % opt.model_save_interval == 0:
		num = str(int(epoch / opt.model_save_interval))
		save_model(discriminator,optimizer_D,epoch,opt.model_save_path+"/"+num+"_D.pt")
		save_model(generator,optimizer_G,epoch,opt.model_save_path+"/"+num+"_G.pt")
	
	# Intermediate plot
	if epoch % (opt.n_epochs/4) == 0:
		#Plot losses			
		plot_losses(G_losses,D_losses)
		#Plot scores
		plot_scores(D_x,D_G_z)
		#Plot began mesure of convergeance
		plot_began(M_plot,k_plot)
	
	print("[Epoch Time: ",time.time()-t_epoch,"s]")

print("[Total Time: ",time.strftime("%Hh:%Mm:%Ss",time.gmtime(time.time()-t_total)),"]")

#Plot losses			
plot_losses(G_losses,D_losses)

#Plot game score
plot_scores(D_x,D_G_z)

#Plot began mesure of convergeance
plot_began(M_plot,k_plot)

# Save model for futur training
save_model(discriminator,optimizer_D,epoch,opt.model_save_path+"/last_D.pt")
save_model(generator,optimizer_G,epoch,opt.model_save_path+"/last_G.pt")

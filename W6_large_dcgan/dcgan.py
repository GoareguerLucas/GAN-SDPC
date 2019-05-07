import argparse
import os
import numpy as np
import math

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
parser.add_argument("-e", "--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lrD", type=float, default=0.0004, help="adam: learning rate for D")
parser.add_argument("--lrG", type=float, default=0.0004, help="adam: learning rate for G")
parser.add_argument("--eps", type=float, default=0.5, help="batchnorm: espilon for numerical stability")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("-i", "--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("-s", "--sample_interval", type=int, default=40, help="interval between image sampling")
parser.add_argument("--sample_path", type=str, default='images')
parser.add_argument("-m", "--model_save_interval", type=int, default=2500, help="interval between image sampling")
parser.add_argument('--model_save_path', type=str, default='models')
parser.add_argument("-d", "--depth", action="store_true", help="Utiliser si utils.py et SimpsonsDataset.py sont deux dossier au dessus.")
opt = parser.parse_args()
print(opt)

# Particular import
depth = ""
if opt.depth == True:
	depth = "../"
sys.path.append(depth+"../")#../../GAN-SDPC/

# Dossier de sauvegarde
#os.makedirs("images", exist_ok=True)
#os.makedirs("model", exist_ok=True)
os.makedirs(opt.sample_path, exist_ok=True)
os.makedirs(opt.model_save_path, exist_ok=True)

cuda = True if torch.cuda.is_available() else False


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


class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		def generator_block(in_filters, out_filters, kernel=5, stride=2):
			block = [nn.ConvTranspose2d(in_filters, out_filters, kernel, stride=stride, padding=1), nn.BatchNorm2d(out_filters, opt.eps), nn.LeakyReLU(0.2, inplace=True)]
			
			return block
		
		self.init_size = opt.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 1024 * self.init_size ** 2), nn.LeakyReLU(0.2, inplace=True))

		self.conv_blocks = nn.Sequential(
			*generator_block(1024, 512, kernel=5, stride=2),
			*generator_block(512, 256, kernel=5, stride=2),
			*generator_block(128, 64, kernel=5, stride=2),
			nn.ConvTranspose2d(64, opt.channels, kernel_size=5, stride=1),
			nn.Tanh(),
		)

	def forward(self, z):
		out = self.l1(z)
		out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
		img = self.conv_blocks(out)
		return img


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		def discriminator_block(in_filters, out_filters, bn=True, kernel=5, stride=2):
			block = [nn.Conv2d(in_filters, out_filters, kernel, stride, 1), nn.LeakyReLU(0.2, inplace=True)]#, nn.Dropout2d(0.25)
			if bn:
				block.append(nn.BatchNorm2d(out_filters, opt.eps))
			return block

		self.model = nn.Sequential(
			*discriminator_block(opt.channels, 64, bn=False),
			*discriminator_block(64, 128),
			*discriminator_block(128, 256),
			*discriminator_block(256, 512, stride=1),
			*discriminator_block(512, 1024),
		)

		# The height and width of downsampled image
		ds_size = opt.img_size // 2 ** 4
		self.adv_layer = nn.Sequential(nn.Linear(8*8*1024, 1))#, nn.Sigmoid()

	def forward(self, img):
		out = self.model(img)
		out = out.view(out.shape[0], -1)
		validity = self.adv_layer(out)

		return validity

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()
#pixelwise_loss = torch.nn.L1Loss()
sigmoid = nn.Sigmoid()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
	generator.cuda()
	discriminator.cuda()
	adversarial_loss.cuda()
	#pixelwise_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = load_data(depth+"../../cropped/cp/",opt.img_size,opt.batch_size)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lrG, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lrD, betas=(opt.b1, opt.b2))

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
d_x_mean = []
d_g_z_mean = []

save_dot = 10 # Nombre d'epochs avant de sauvegarder un point des courbes
batch_on_save_dot = save_dot*len(dataloader)

# Vecteur z fixe pour faire les samples 
fixed_noise = Variable(Tensor(np.random.normal(0, 1, (25, opt.latent_dim))))

t_total = time.time()
for epoch in range(1,opt.n_epochs+1):
	t_epoch = time.time()
	#scheduler_G.step()
	#scheduler_D.step()
	for i, (imgs, _) in enumerate(dataloader):
		t_batch = time.time()
		
		# Adversarial ground truths
		#valid_smooth = Variable(Tensor(imgs.shape[0], 1).fill_(float(np.random.uniform(0.9, 1.0, 1))), requires_grad=False)
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
		# Ajout d'un bruit au image  réels 
		#rand = Tensor(imgs.shape).normal_(0.0, 0.1)
		#Discriminator descision
		d_x = discriminator(real_imgs)
		# Measure discriminator's ability to classify real from generated samples
		real_loss = adversarial_loss(d_x, valid)
		# Backward
		real_loss.backward()
		
		# Fake batch
		#Discriminator descision
		d_g_x = discriminator(gen_imgs.detach())
		# Measure discriminator's ability to classify real from generated samples
		fake_loss = adversarial_loss(d_g_x, fake)
		# Backward
		fake_loss.backward()
		
		d_loss = real_loss + fake_loss

		optimizer_D.step()
		
		# -----------------
		#  Train Generator
		# -----------------

		optimizer_G.zero_grad()
		
		# New discriminator descision, Since we just updated D
		d_g_x = discriminator(gen_imgs)
		# Loss measures generator's ability to fool the discriminator
		g_loss = adversarial_loss(d_g_x, valid)
		# Backward
		g_loss.backward()
		
		optimizer_G.step()


		print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time: %fs]"
			% (epoch, opt.n_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item(), time.time()-t_batch)
		)
		
		# Compensation pour le BCElogits
		d_x = sigmoid(d_x)
		d_g_x = sigmoid(d_g_x)
		
		# Save Losses and scores for plotting later
		g_losses.append(g_loss.item())
		d_losses.append(d_loss.item())
		d_x_mean.append(torch.sum(d_x).item()/imgs.size(0))
		d_g_z_mean.append(torch.sum(d_g_x).item()/imgs.size(0))
		
	# Save samples
	if epoch % opt.sample_interval == 0:
		sampling(fixed_noise, generator, opt.sample_path, epoch)
	
	# Save Losses and scores for plotting later
	if epoch % save_dot == 0:
		G_losses.append(sum(g_losses)/batch_on_save_dot)
		D_losses.append(sum(d_losses)/batch_on_save_dot)
		g_losses = []
		d_losses = []
		D_x.append(sum(d_x_mean)/batch_on_save_dot)
		D_G_z.append(sum(d_g_z_mean)/batch_on_save_dot)
		d_x_mean = []
		d_g_z_mean = []
	
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
	
	print("[Epoch Time: ",time.time()-t_epoch,"s]")

print("[Total Time: ",time.strftime("%Hh:%Mm:%Ss",time.gmtime(time.time()-t_total)),"]")

#Plot losses			
plot_losses(G_losses,D_losses)

#Plot game score
plot_scores(D_x,D_G_z)

# Save model for futur training
save_model(discriminator,optimizer_D,epoch,opt.model_save_path+"/last_D.pt")
save_model(generator,optimizer_G,epoch,opt.model_save_path+"/last_G.pt")

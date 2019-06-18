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
import random

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lrG", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--lrD", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--eps", type=float, default=0.00005, help="batchnorm: espilon for numerical stability")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval betwen image samples")
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

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		def block(in_feat, out_feat, normalize=True):
			layers = [nn.Linear(in_feat, out_feat)]
			if normalize:
				layers.append(nn.BatchNorm1d(out_feat, opt.eps))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*block(opt.latent_dim, 128, normalize=False),
			*block(128, 256),
			*block(256, 512),
			*block(512, 1024),
			nn.Linear(1024, int(np.prod(img_shape))),
			nn.Tanh()
		)

	def forward(self, z):
		img = self.model(z)
		img = img.view(img.size(0), *img_shape)
		return img


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(int(np.prod(img_shape)), 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid(),
		)

	def forward(self, img):
		img_flat = img.view(img.size(0), -1)
		validity = self.model(img_flat)

		return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
	generator.cuda()
	discriminator.cuda()
	adversarial_loss.cuda()

# Configure data loader
dataloader = load_data("../../cropped/cp/",opt.img_size,opt.batch_size)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lrG, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lrD, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

# Liste de sauvegarde pour les courbes
G_losses = []
D_losses = []
g_losses = []
d_losses = []
D_x = []
D_G_z = []
d_x = []
d_g_z = []

save_dot = 1 # Nombre d'epochs avant de sauvegarder un point des courbes
batch_on_save_dot = save_dot*len(dataloader)

t_total = time.time()
for epoch in range(1,opt.n_epochs+1):
	t_epoch = time.time()
	for i, (imgs, _) in enumerate(dataloader):
		t_batch = time.time()
		
		# Adversarial ground truths
		valid_smooth = Variable(Tensor(imgs.shape[0], 1).fill_(float(np.random.uniform(0.9, 1.0, 1))), requires_grad=False)
		valid = Variable(Tensor(imgs.size(0), 1).fill_(1), requires_grad=False)
		fake = Variable(Tensor(imgs.size(0), 1).fill_(0), requires_grad=False)

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
		g_loss = adversarial_loss(discriminator(gen_imgs), valid)

		g_loss.backward()
		optimizer_G.step()

		# ---------------------
		#  Train Discriminator
		# ---------------------

		optimizer_D.zero_grad()
		
		#Discriminator descision
		d_x_tmp = discriminator(real_imgs)
		d_g_x_tmp = discriminator(gen_imgs.detach())
		
		# Measure discriminator's ability to classify real from generated samples
		real_loss = adversarial_loss(d_x_tmp, valid_smooth)
		fake_loss = adversarial_loss(d_g_x_tmp, fake)
		
		d_loss = (real_loss + fake_loss) / 2

		d_loss.backward()
		optimizer_D.step()

		print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time: %fs]"
			% (epoch, opt.n_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item(), time.time()-t_batch)
		)
	
		# Save Losses and scores for plotting later
		g_losses.append(g_loss.item())
		d_losses.append(d_loss.item())
		d_x.append(torch.sum(d_x_tmp).item()/imgs.size(0))
		d_g_z.append(torch.sum(d_g_x_tmp).item()/imgs.size(0))
		
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

"""#Load model
g = Generator()
d = Discriminator()

optimizer_g = torch.optim.Adam(g.parameters(), lr=opt.lrG, betas=(opt.b1, opt.b2))
optimizer_d = torch.optim.Adam(d.parameters(), lr=opt.lrD, betas=(opt.b1, opt.b2))

print(load_model(d,optimizer_d,opt.model_save_path+"/last_D.pt"))
print(load_model(g,optimizer_g,opt.model_save_path+"/last_G.pt"))

print(g)
print("---------------------------\n")
print(d)

g.train()
d.train()"""

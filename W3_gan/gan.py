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
import matplotlib.pyplot as plt
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lrG", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--lrD", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--eps", type=float, default=0.00005, help="batchnorm: espilon for numerical stability")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=4000, help="interval betwen image samples")
parser.add_argument("--sample_path", type=str, default='images')
parser.add_argument("--model_save_interval", type=int, default=100000, help="interval between image sampling")
parser.add_argument('--model_save_path', type=str, default='model')
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
transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
dataset = FastSimpsonsDataset("/var/lib/vz/data/g14006889/cropped/cp/",opt.img_size,opt.img_size,transformations) #../../../Dataset/cropped/
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

"""os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
	datasets.MNIST(
		"../../data/mnist",
		train=True,
		download=True,
		transform=transforms.Compose(
			[transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
		),
	),
	batch_size=opt.batch_size,
	shuffle=True,
)"""

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

t_total = time.time()
for epoch in range(opt.n_epochs):
	t_epoch = time.time()
	for i, (imgs, _) in enumerate(dataloader):
		t_batch = time.time()
		
		# Adversarial ground truths
		valid = Variable(Tensor(imgs.size(0), 1).fill_(float(np.random.uniform(0.8, 1.0, 1))), requires_grad=False)
		fake = Variable(Tensor(imgs.size(0), 1).fill_(0), requires_grad=False)
		#fake = Variable(Tensor(imgs.size(0), 1).fill_(float(np.random.uniform(0.0, 0.3, 1))), requires_grad=False)
		#valid = Variable(Tensor(imgs.size(0), 1).uniform_(0.4, 1.0), requires_grad=False)
		#fake = Variable(Tensor(imgs.size(0), 1).uniform_(0.0, 0.6), requires_grad=False)

		# Configure input
		#print("In ",imgs.shape)
		#print(imgs.data[0,0,0,0])
		
		#print(Tensor(np.random.normal(0.0, 0.05, (imgs.shape[0], opt.channels, opt.img_size, opt.img_size))).data[0,0,0,0])
		#rand = Tensor(imgs.shape).normal_(0.0, 0.15)
		#print(rand[0,0,0,0])
		
		#real_imgs = imgs.type(Tensor) + rand
		real_imgs = Variable(imgs.type(Tensor))
		#print(real_imgs[0,0,0,0])
		#print(real_imgs.shape)
		
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

		# Measure discriminator's ability to classify real from generated samples
		"""if random.randint(1,6) == 3:
			real_loss = adversarial_loss(discriminator(real_imgs), fake)
			fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), valid)
		else:"""
		real_loss = adversarial_loss(discriminator(real_imgs), valid)
		fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
		
		d_loss = (real_loss + fake_loss) / 2

		d_loss.backward()
		optimizer_D.step()

		print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time: %fs]"
			% (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), time.time()-t_batch)
		)

		batches_done = epoch * len(dataloader) + i
		if batches_done % opt.sample_interval == 0:
			save_image(gen_imgs.data[:25], "%s/%d.png" % (opt.sample_path, batches_done), nrow=5, normalize=True)
		
		
		# Save Losses for plotting later
		g_losses.append(g_loss.item())
		d_losses.append(d_loss.item())
		#print("INFO : ",sum(discriminator(real_imgs.detach())).item()/imgs.size(0))
		d_x.append(sum(discriminator(real_imgs.detach())).item()/imgs.size(0))
		d_g_z.append(sum(discriminator(gen_imgs.detach())).item()/imgs.size(0))
		if batches_done % 100 == 0:
			G_losses.append(sum(g_losses)/100)
			D_losses.append(sum(d_losses)/100)
			g_losses = []
			d_losses = []
			D_x.append(sum(d_x)/100)
			D_G_z.append(sum(d_g_z)/100)
			d_x = []
			d_g_z = []
			
		if batches_done % opt.model_save_interval == 0:
			num = str(int(batches_done / opt.model_save_interval))
			torch.save(discriminator, opt.model_save_path+"/"+num+"_D.pt")
			torch.save(generator, opt.model_save_path+"/"+num+"_G.pt")
	
	print("[Epoch Time: ",time.time()-t_epoch,"s]")

print("[Total Time: ",time.time()-t_total,"s]")

torch.save(discriminator, opt.model_save_path+"/last_D.pt")
torch.save(generator, opt.model_save_path+"/last_G.pt")

"""G = torch.load("G.pt")
D = torch.load("D.pt")
print(G)
print(D)"""


#Plot losses			
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(opt.sample_path+"/losses",format="png")
#plt.show()

#Plot game score
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator scores During Training")
plt.plot(D_x,label="D(x)")
plt.plot(D_G_z,label="D(G(z))")
plt.xlabel("iterations")
plt.ylabel("scores")
plt.legend()
plt.savefig(opt.sample_path+"/scores",format="png")

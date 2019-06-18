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
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=4000, help="interval between image sampling")
parser.add_argument("--sample_path", type=str, default='images')
parser.add_argument("--model_save_interval", type=int, default=40000, help="interval between image sampling")
parser.add_argument('--model_save_path', type=str, default='models')
opt = parser.parse_args()
print(opt)

# Dossier de sauvegarde
#os.makedirs("images", exist_ok=True)
#os.makedirs("model", exist_ok=True)
os.makedirs(opt.sample_path, exist_ok=True)
os.makedirs(opt.model_save_path, exist_ok=True)

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
			nn.BatchNorm2d(128, opt.eps),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, opt.eps),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
			nn.Tanh(),
		)

	def forward(self, z):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks(out)
		return img


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		def discriminator_block(in_filters, out_filters, bn=True):
			block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
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
		ds_size = opt.img_size // 2 ** 4
		self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

	def forward(self, img):
		out = self.model(img)
		out = out.view(out.shape[0], -1)
		validity = self.adv_layer(out)

		return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()
#pixelwise_loss = torch.nn.L1Loss()

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
transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
dataset = FastSimpsonsDataset("../../cropped/cp/",opt.img_size,opt.img_size,transformations) #../../../Dataset/cropped/
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

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
		valid_smooth = Variable(Tensor(imgs.shape[0], 1).fill_(float(np.random.uniform(0.7, 1.0, 1))), requires_grad=False)
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
		"""g_loss = 0.1 * adversarial_loss(discriminator(gen_imgs), valid) + 0.9 * pixelwise_loss(
			gen_imgs, real_imgs
		)"""

		g_loss.backward()
		optimizer_G.step()

		# ---------------------
		#  Train Discriminator
		# ---------------------

		optimizer_D.zero_grad()

		# Measure discriminator's ability to classify real from generated samples
		rand = Tensor(imgs.shape).normal_(0.0, 0.25)
		d_x_tmp = discriminator(real_imgs + rand)
		d_g_x_tmp = discriminator(gen_imgs.detach() + rand)
		
		real_loss = adversarial_loss(d_x_tmp, valid_smooth)
		fake_loss = adversarial_loss(d_g_x_tmp, fake)
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
		d_x.append(sum(d_x_tmp).item()/imgs.size(0))
		d_g_z.append(sum(d_g_x_tmp).item()/imgs.size(0))
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
			torch.save(discriminator.state_dict(), opt.model_save_path+"/last_D.pt")
			torch.save(generator.state_dict(), opt.model_save_path+"/last_G.pt")

	print("[Epoch Time: ",time.time()-t_epoch,"s]")

print("[Total Time: ",time.time()-t_total,"s]")
			
torch.save(discriminator.state_dict(), opt.model_save_path+"/last_D.pt")
torch.save(generator.state_dict(), opt.model_save_path+"/last_G.pt")

"""
G = Generator()
G = torch.load_state_dict(torch.load("G.pt"))
D = Discriminator()
D = torch.load_state_dict(torch.load("D.pt"))
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
plt.savefig("losses.png",format="png")
#plt.show()

#Plot game score
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator scores During Training")
plt.plot(D_x,label="D(x)")
plt.plot(D_G_z,label="D(G(z))")
plt.xlabel("iterations")
plt.ylabel("scores")
plt.legend()
plt.savefig("scores.png",format="png")

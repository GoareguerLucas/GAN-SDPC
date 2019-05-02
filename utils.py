from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import datetime
import numpy as np
import random
import torch

import sys
sys.path.append("../")#../../GAN-SDPC/

from SimpsonsDataset import SimpsonsDataset,FastSimpsonsDataset

def load_data(path,img_size,batch_size,Fast=True):
	transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

	if Fast:
		dataset = FastSimpsonsDataset(path, img_size, img_size, transformations) #../../../Dataset/
	else:
		dataset = SimpsonsDataset(path, img_size, img_size, transformations) #../../../Dataset/

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

	return dataloader

def save_model(model, optimizer, epoch, path):
	info = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		}
	torch.save(info,path)

def load_model(model, optimizer, path):
	checkpoint = torch.load(path)

	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	return checkpoint['epoch']

def plot_scores(D_x,D_G_z):
	#Plot game score
	fig = plt.figure(figsize=(10,5))
	plt.title("Generator and Discriminator scores During Training")
	plt.plot(D_x,label="D(x)")
	plt.plot(D_G_z,label="D(G(z))")
	plt.xlabel("Epochs (/10)")
	plt.ylabel("scores")
	plt.legend()
	plt.ylim([0.0,1.1])
	plt.savefig("scores.png",format="png")
	plt.close(fig)

def plot_losses(G_losses,D_losses):
	#Plot losses
	fig = plt.figure(figsize=(10,5))
	plt.title("Generator and Discriminator Loss During Training")
	plt.plot(G_losses,label="G")
	plt.plot(D_losses,label="D")
	plt.xlabel("Epochs (/10)")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig("losses.png",format="png")
	plt.close(fig)

def plot_began(M,k):
	#Plot M and k value
	fig = plt.figure(figsize=(10,5))
	plt.title("M and k Value During Training")
	plt.plot(M,label="M")
	plt.plot(k,label="k")
	plt.xlabel("Epochs (/10)")
	plt.ylabel("Value")
	plt.legend()
	plt.ylim([0.0,1.1])
	plt.savefig("M_k.png",format="png")
	plt.close(fig)

def plot_lr(lr):
	#Plot lr
	fig = plt.figure(figsize=(10,5))
	plt.title("lr Value During Training")
	plt.plot(lr,label="lr")
	plt.xlabel("Epochs (/10)")
	plt.ylabel("Value")
	plt.legend()
	plt.savefig("lr.png",format="png")
	plt.close(fig)

if __name__ == "__main__":

	D_G_z = np.random.normal(0.5,0.5,100)
	D_x = np.random.normal(0.5,0.5,100)

	plot_scores(D_x,D_G_z)

	print("test")

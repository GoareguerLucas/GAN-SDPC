from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import datetime
import numpy as np
import random
import torch
import imageio
import time


import sys
sys.path.append("../")#../../GAN-SDPC/

from SimpsonsDataset import SimpsonsDataset,FastSimpsonsDataset

def load_data(path,img_size,batch_size,Fast=True,rand_hflip=False,return_dataset=False):
	print("Loading data...")
	t_total = time.time()
	
	transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

	if Fast:
		dataset = FastSimpsonsDataset(path, img_size, img_size, transformations, rand_hflip) #../../../Dataset/
	else:
		dataset = SimpsonsDataset(path, img_size, img_size, transformations) #../../../Dataset/

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	
	print("[Loading Time: ",time.strftime("%Mm:%Ss",time.gmtime(time.time()-t_total)),"]\n")
	
	if return_dataset == True:
		return dataloader, dataset
	return dataloader

def save_model(model, optimizer, epoch, path):
	print("Save model : ",model._name())
	info = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		}
	torch.save(info,path)

def load_model(model, optimizer, path):
	print("Load model :",model._name())
	checkpoint = torch.load(path)

	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	return checkpoint['epoch']

def load_models(discriminator,optimizer_D,generator,optimizer_G,n_epochs,model_save_path):
	start_epochD = load_model(discriminator, optimizer_D, model_save_path+"/last_D.pt")
	start_epochG = load_model(generator, optimizer_G, model_save_path+"/last_G.pt")
	
	if start_epochG is not start_epochD:
		print("Error : G trained different times of D  !!")
		exit(0)
	start_epoch = start_epochD
	if start_epoch >= n_epochs:
		print("Error : Nombre d'epochs demander inférieur au nombre d'epochs déjà effectuer !!")
		exit(0)
		
	return start_epoch+1 # La dernière epoch est déjà faite

def plot_scores(D_x,D_G_z,start_epoch=1,current_epoch=-1):
	if len(D_x) <= 0 or len(D_G_z) <= 0:
		return None
	
	if current_epoch == -1: # C'est pour surcharger la fonction pour les versions passer
		current_epoch = len(D_x)*10
	
	#Plot game score
	fig = plt.figure(figsize=(10,5))
	plt.title("Generator and Discriminator scores During Training")
	plt.plot(D_x,label="D(x)")
	plt.plot(D_G_z,label="D(G(z))")
	plt.xlabel("Epochs")
	plt.ylabel("Scores")
	plt.legend()
	# Gradutation
	plt.yticks(np.arange(0.0,1.2,0.1))
	positions = np.linspace(0,len(D_x),num=6)
	labels = np.linspace(start_epoch-1,current_epoch,num=6)
	plt.xticks(positions, labels)
	
	plt.grid(True)
	plt.savefig("scores.png",format="png")
	plt.close(fig)

def plot_losses(G_losses,D_losses,start_epoch=1,current_epoch=-1):
	if len(G_losses) <= 0 or len(D_losses) <= 0:
		return None
	
	if current_epoch == -1: # C'est pour surcharger la fonction pour les versions passer
		current_epoch = len(D_losses)*10
	
	#Plot losses
	fig = plt.figure(figsize=(10,5))
	plt.title("Generator and Discriminator Loss During Training")
	plt.plot(G_losses,label="G")
	plt.plot(D_losses,label="D")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	# Gradutation
	positions = np.linspace(0,len(D_losses),num=6)
	labels = np.linspace(start_epoch-1,current_epoch,num=6)
	plt.xticks(positions, labels)
	
	plt.grid(True)
	plt.savefig("losses.png",format="png")
	plt.close(fig)

def plot_began(M,k,start_epoch=1,current_epoch=-1):
	if len(M) <= 0 or len(k) <= 0:
		return None
		
	if current_epoch == -1: # C'est pour surcharger la fonction pour les versions passer
		current_epoch = len(M)*10
	
	#Plot M and k value
	fig = plt.figure(figsize=(10,5))
	plt.title("M and k Value During Training")
	plt.plot(M,label="M")
	plt.plot(k,label="k")
	plt.xlabel("Epochs")
	plt.ylabel("Value")
	plt.legend()
	# Gradutation
	plt.yticks(np.arange(0.0,1.2,0.1))
	positions = np.linspace(0,len(M),num=6)
	labels = np.linspace(start_epoch-1,current_epoch,num=6)
	plt.xticks(positions, labels)
	
	plt.grid(True)
	plt.savefig("M_k.png",format="png")
	plt.close(fig)

def plot_lr(lr,start_epoch=1,current_epoch=-1):
	if len(lr) <= 0:
		return None
		
	if current_epoch == -1: # C'est pour surcharger la fonction pour les versions passer
		current_epoch = len(lr)*10
	
	#Plot lr
	fig = plt.figure(figsize=(10,5))
	plt.title("lr Value During Training")
	plt.plot(lr,label="lr")
	plt.xlabel("Epochs")
	plt.ylabel("Value")
	plt.legend()
	# Gradutation
	positions = np.linspace(0,len(lr),num=6)
	labels = np.linspace(start_epoch-1,current_epoch,num=6)
	plt.xticks(positions, labels)
	
	plt.grid(True)
	plt.savefig("lr.png",format="png")
	plt.close(fig)

"""
Utilise generator et noise pour générer une images sauvegarder à path/epoch.png
Le sample est efféctuer en mode eval pour generator puis il est de nouveau régler en mode train.
"""
def sampling(noise, generator, path, epoch):
	generator.eval()
	gen_imgs = generator(noise)
	save_image(gen_imgs.data[:], "%s/%d.png" % (path, epoch), nrow=5, normalize=True)
	generator.train()

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)
	print()

def comp(s):
	s = s.split("/")[-1] # Nom du fichier
	num = s.split(".")[0] # Numéro dans le nom du fichier
	
	return int(num)

def generate_animation(path):
	images_path = glob(path + '[0-9]*.png')
	
	images_path = sorted(images_path,key=comp)
	
	images = []
	for i in images_path:
		#print(i)
		images.append(imageio.imread(i))
	imageio.mimsave(path + 'training.gif', images, fps=1)

def histogram(D_x,D_G_z,epoch):
	fig = plt.figure(figsize=(10,5))
	plt.title("D(x) réponse pour l'epochs "+str(epoch))
	plt.hist(D_x,bins=16)
	plt.scatter(D_x,np.zeros(64),s=60,color='r',marker="|")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	
	plt.savefig("hist/dx_"+str(epoch)+".png",format="png")
	plt.close(fig)

	fig = plt.figure(figsize=(10,5))
	plt.title("D(G(z)) réponse pour l'epochs "+str(epoch))
	plt.hist(D_G_z,bins=16)
	plt.scatter(D_G_z,np.zeros(64),s=60,color='r',marker="|")
	plt.xlabel("Value")
	plt.ylabel("Frequency")

	plt.savefig("hist/dgz_"+str(epoch)+".png",format="png")
	plt.close(fig)

def plot_extrem(D_x,D_G_z,nb_batch,start_epoch=1,current_epoch=-1,name="extremum.png"):
	if len(D_x) <= 0 or len(D_G_z) <= 0:
		return None
	
	#Plot D_x and D_x value
	fig = plt.figure(figsize=(10,5))
	plt.title("Extrem response of D During Training")
	plt.plot(D_x,label="Log10(D_x.min())")
	plt.plot(D_G_z,label="Log10(D_G_z.min())")
	plt.xlabel("Epochs")
	plt.ylabel("Value")
	plt.legend()
	# Gradutation
	positions = np.linspace(0,current_epoch*nb_batch,num=6)
	labels = np.linspace(start_epoch-1,current_epoch,num=6)
	plt.xticks(positions, labels)
	
	plt.grid(True)
	plt.savefig(name,format="png")
	plt.close(fig)

if __name__ == "__main__":

	"""D_G_z = np.random.normal(0.5,0.5,100)
	D_x = np.random.normal(0.5,0.5,100)

	plot_scores(D_x,D_G_z)

	print("test")"""
	
	#generate_animation("W7_128_dcgan/gif/")

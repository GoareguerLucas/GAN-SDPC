from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision  
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import datetime
import numpy as np
import random

INPUT_DATA_DIR = "../cropped/cp/"
IMAGE_SIZE = 200
OUTPUT_DIR = './{date:%Y-%m-%d_%H:%M:%S}/'.format(date=datetime.datetime.now())

def show_tensor(sample_tensor, epoch):
	#figure, axes = plt.subplots(1, len(sample_tensor), figsize = (IMAGE_SIZE, IMAGE_SIZE))
	#for index, axis in enumerate(axes):
	#	axis.axis('off')
	
	tensor_view = sample_tensor.permute(1, 2, 0)

	print(tensor_view.shape)
	
	plt.imshow(np.asarray(tensor_view))
		
	plt.show()


class SimpsonsDataset(Dataset):
	def __init__(self, dir_path, height, width, transforms=None):
		"""
		Args:
			dir_path (string): path to dir conteint exclusively images png
			height (int): image height
			width (int): image width
			transform: pytorch transforms for transforms and tensor conversion
		"""
		self.files = glob(dir_path + '*')
		self.labels = np.zeros(len(self.files))
		self.height = height
		self.width = width
		self.transforms = transforms

	def __getitem__(self, index):
		single_image_label = self.labels[index]
		# Read each pixels and reshape the 1D array to 2D array
		img_as_np = np.asarray(Image.open(self.files[index]).resize((self.height, self.width))).astype('uint8')
		# Convert image from numpy array to PIL image
		img_as_img = Image.fromarray(img_as_np)
		img_as_img = img_as_img.convert('RGB')
		# Transform image to tensor
		if self.transforms is not None:
			img_as_tensor = self.transforms(img_as_img)
		# Return image and the label
		return (img_as_tensor, single_image_label)

	def __len__(self):
		return len(self.files)
		
class FastSimpsonsDataset(Dataset):
	def __init__(self, dir_path, height, width, transforms=None, rand_hflip=False, mode="RGB"):
		"""
		Args:
			dir_path (string): path to dir conteint exclusively images png
			height (int): image height
			width (int): image width
			transform: pytorch transforms for transforms and tensor conversion
		"""
		self.files = glob(dir_path + '*')
		self.labels = np.zeros(len(self.files))
		self.height = height
		self.width = width
		self.transforms = transforms
		self.rand_hflip = rand_hflip
		
		# Chargement des images
		self.tensors = list()
		for img in self.files:
			img_as_np = np.asarray(Image.open(img).resize((self.height, self.width))).astype('uint8')
			# Convert image from numpy array to PIL image
			img_as_img = Image.fromarray(img_as_np)
			img_as_img = img_as_img.convert('RGB')
				
			# Transform image to tensor
			if self.transforms is not None:
				img_as_tensor = self.transforms(img_as_img)
			
			if mode == "HSV":
				img_as_img = torchvision.transforms.ToPILImage()(img_as_tensor)
				HSV = img_as_img.convert('HSV')
				img_as_tensor = torchvision.transforms.ToTensor(HSV)
			
			self.tensors.append(img_as_tensor)

	def __getitem__(self, index):
		single_image_label = self.labels[index]
		img_as_tensor = self.tensors[index]
		
		if self.rand_hflip:
			img_as_tensor = img_as_tensor.flip(2)

		# Return image and the label
		return (img_as_tensor, single_image_label)

	def __len__(self):
		return len(self.files)
		

if __name__ == "__main__":
	
	
	transformations = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
	simpsonsDataset = SimpsonsDataset(INPUT_DATA_DIR, IMAGE_SIZE, IMAGE_SIZE, transformations)
	
	print(type(simpsonsDataset), len(simpsonsDataset))
	
	nb_images = 100
	
	images = []
	for i in range(nb_images):
		images.append(np.asarray(simpsonsDataset.__getitem__(i)[0].permute(1, 2, 0)))
	images = np.asarray(images)
	print(images.shape)
	
	bags = np.reshape(images,(-1,3))
	print("Bags shape :",bags.shape)
	print(bags[0])
	
	from torchvision.utils import save_image
	import torch
	
	batch_size= 4
	batch = []
	for i in range(batch_size):
		batch.append(np.asarray(simpsonsDataset.__getitem__(i)[0]))
	batch = torch.tensor(batch)
	print(batch.shape)
	
	#print("Shape image :",real.shape)
	#save_image(real, "test.png", nrow=1, normalize=True)
	
	taux = 0.1 # Part des pixels de i à remplacer
	
	nb_pixels = int(batch_size*IMAGE_SIZE*IMAGE_SIZE * taux)
	print("Pixel bruiter par batch :",nb_pixels)
	
	idx = np.random.permutation(np.arange(len(bags)))[:nb_pixels]
	print("Index du bruit dans bags:")
	print(idx.shape)
	#print(idx)
	
	pixels = bags[idx]
	print("Pixels bruitées choisis")
	print(pixels.shape)
	#print(pixels)
	
	nb = IMAGE_SIZE*IMAGE_SIZE
	nb_by_batch = batch_size*IMAGE_SIZE**2
	print("Pixels par images :",nb)
	print("Pixels par batch :",nb_by_batch)
	
	# Construction
	mask = np.ones((nb_by_batch,3))
	print("Mask shape :",mask.shape)
	noise = np.zeros((nb_by_batch,3))
	print("Noise shape :",noise.shape)
	
	# Random pixels idx
	pixels_idx = np.random.permutation(np.arange(batch_size*IMAGE_SIZE**2))[:nb_pixels]
	print("Pixels idx shape :",pixels_idx.shape)
	
	# Remplissage
	mask[pixels_idx] = 0
	noise[pixels_idx] += pixels
	print("mask")
	print(mask)
	print("noise")
	print(noise)
	 
	# Reshape (IMAGE_SIZE**2,3)=>(IMAGE_SIZE,IMAGE_SIZE,3) / Transpose (IMAGE_SIZE,IMAGE_SIZE,3)=>(3,IMAGE_SIZE,IMAGE_SIZE)
	mask = mask.reshape((batch_size,IMAGE_SIZE,IMAGE_SIZE,3)).transpose(0,-1,1,2)
	noise = noise.reshape((batch_size,IMAGE_SIZE,IMAGE_SIZE,3)).transpose(0,-1,1,2)
	print("Noise shape :",noise.shape)
	print("Mask shape :",mask.shape)
	
	# Ajout dans l'image
	clear_image = batch * torch.tensor(mask).float()
	noised_image = clear_image + torch.tensor(noise).float()
	
	print(batch[1][1])
	
	print(clear_image[1][1])
	
	print(noised_image[1][1])
	
	save_image(noised_image, "test_fin.png", nrow=1, normalize=False)
	 
	#show_tensor(item[0],0)

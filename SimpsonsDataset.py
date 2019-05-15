from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import datetime
import numpy as np
import random

INPUT_DATA_DIR = "../Dataset/cropped/"
IMAGE_SIZE = 32
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
	def __init__(self, dir_path, height, width, transforms=None, rand_hflip=False):
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
			self.tensors.append(img_as_tensor)

	def __getitem__(self, index):
		single_image_label = self.labels[index]
		img_as_tensor = self.tensors[index]
		print (img_as_tensor)
		if self.rand_hflip:
			img_as_tensor = img_as_tensor.flip(2)
		print (img_as_tensor)
		# Return image and the label
		return (img_as_tensor, single_image_label)

	def __len__(self):
		return len(self.files)
		

if __name__ == "__main__":
	
	
	transformations = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
	simpsonsDataset = SimpsonsDataset(INPUT_DATA_DIR, IMAGE_SIZE, IMAGE_SIZE, transformations)
	
	print(type(simpsonsDataset), len(simpsonsDataset))
	
	item = simpsonsDataset.__getitem__(1)
	print(type(item[0]), item[0].shape)
	
	
	show_tensor(item[0],0)


	

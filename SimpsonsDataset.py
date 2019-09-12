from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import datetime
import numpy as np
import random
import torch
try:
    from skimage.color import rgb2hsv
except:
    pass


def show_tensor(sample_tensor, epoch):
    #figure, axes = plt.subplots(1, len(sample_tensor), figsize = (IMAGE_SIZE, IMAGE_SIZE))
    # for index, axis in enumerate(axes):
    #   axis.axis('off')

    tensor_view = sample_tensor.permute(1, 2, 0)

    print(tensor_view.shape)

    plt.imshow(np.asarray(tensor_view))

    plt.show()


class SimpsonsDataset(Dataset):
    def __init__(self, dir_path, height, width, transforms=None, mode="RGB"):
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
        self.mode = mode

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Read each pixels and reshape the 1D array to 2D array
        img_as_np = np.asarray(Image.open(self.files[index]).resize((self.height, self.width))).astype('uint8')
        # Convert image from numpy array to PIL image
        #img_as_img = Image.fromarray(img_as_np)
        #img_as_img = img_as_img.convert('RGB')

        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_np)

        # Use HSV format
        if self.mode == "HSV":
            array = img_as_tensor.permute(2, 1, 0).numpy()
            # print(array.shape)
            HSV = rgb2hsv(array)
            # print(HSV.shape)
            img_as_tensor = torch.from_numpy(array).permute(2, 1, 0)
            # print(img_as_tensor.shape)

        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.files)


class FastSimpsonsDataset(Dataset):
    def __init__(self, dir_path, height, width, transform, mode="RGB"):
        """
        Args:
                dir_path (string): path to dir conteint exclusively images png
                height (int): image height
                width (int): image width
                transform: pytorch transforms for transforms and tensor conversion during training
        """
        self.files = glob(dir_path + '*')
        self.labels = np.zeros(len(self.files))
        self.height = height
        self.width = width
        self.transform = transform
        self.mode = mode

        # Chargement des images
        self.imgs = list()
        for img in self.files:

            #img_as_np = np.asarray(Image.open(img).resize((self.height, self.width))).astype('uint8')
            img_as_img = Image.open(img).resize((self.height, self.width))

            # Use HSV format
            if self.mode == "HSV":
                array = np.asarray(img).astype('uint8')
                # print(array.shape)
                HSV_array = rgb2hsv(array)
                img_as_img = Image.fromarray(HSV_array, mode="HSV")
                # print(HSV.shape)
                #img_as_tensor = torch.from_numpy(array).permute(2, 1, 0)
                # print(img_as_tensor.shape)

            self.imgs.append(img_as_img)

    def __getitem__(self, index):
        #print("Image load : ",self.files[index])
        single_image_label = self.labels[index]
        img_as_img = self.imgs[index]

        # Transform image to tensor
        img_as_tensor = self.transform(img_as_img)
        
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.files)

class FastFDD(Dataset):
    def __init__(self, dir_path, height, width, transform):
        """
        Args:
                dir_path (string): path to dir conteint exclusively images png
                height (int): image height
                width (int): image width
                transform: pytorch transforms for transforms and tensor conversion during training
        """
        self.files = glob(dir_path + '*')
        self.labels = np.zeros(len(self.files))
        self.height = height
        self.width = width
        self.transform = transform

        # Chargement des images
        self.imgs = list()
        for img in self.files:
            img_as_img = Image.open(img).resize((self.height, self.width))
            self.imgs.append(img_as_img)
            
        # Chargement des noms des images
        self.vectors = list()
        # Mise en forme du nom des fichiers
        for path in self.files:
            path = path[len(dir_path):] # Supp path dir
            path = path[:-4] # Supp extension
            vector = path.replace('_',' ').split()
            vector = [float(e) for e in vector]
            vector = torch.tensor(vector,dtype=torch.float64)
            print(vector)
            print(vector[0].item())
            self.vectors.append(vector)
                

    def __getitem__(self, index):
        #print("Image load : ",self.files[index])
        single_image_label = self.labels[index]
        vector = self.vectors[index]
        img_as_img = self.imgs[index]

        # Transform image to tensor
        img_as_tensor = self.transform(img_as_img)
        
        # Return image and the label
        return (img_as_tensor, single_image_label, vector)

    def __len__(self):
        return len(self.files)

class FastClassifiedDataset(Dataset):
    def __init__(self, dir_path, height, width, transform_constante=None, transform_tmp=None,):
        """
        Args:
                dir_path (string): path to dir conteint exclusively images png
                height (int): image height
                width (int): image width
                transform_constante: pytorch transforms for transforms and tensor conversion before training
                transform_tmp: pytorch transforms for transforms and tensor conversion during training
        """
        self.dir = glob(dir_path + '*')
        
        self.total = 0
        
        # Lecture du noms des images
        self.classes = list()
        for d in self.dir:
            self.classes.append(glob(d + '/*'))
            self.total = self.total + len(self.classes[-1])
            
        for i,c in enumerate(self.classes):
            print("Files in class ",i,": ",len(c))
        print("Dir : ",len(self.dir))
        print(self.dir)
        
        # Construction des labels
        nb_label = 0
        self.labels = np.zeros(self.total)
        for i in range(1,len(self.classes)):
            c_length = len(self.classes[i])
            nb_label = nb_label + len(self.classes[i-1])
            self.labels[nb_label:nb_label+c_length] = i
            
        #u,c = np.unique(self.labels,return_counts=True)
        #print(u)
        #print(c)
        
        self.height = height
        self.width = width
        self.transform_constante = transform_constante
        self.transform_tmp = transform_tmp

        # Chargement des images
        self.imgs = list()
        for c in self.classes:
            for img in c:
                #img_as_np = np.asarray(Image.open(img).resize((self.height, self.width))).astype('uint8')
                img_as_img = Image.open(img).resize((self.height, self.width))
                # Convert image from numpy array to PIL image
                #img_as_img = Image.fromarray(img_as_np)
                #img_as_img = img_as_img.convert('RGB')

                # Transform image to tensor
                if self.transform_constante is not None:
                  img_as_img = self.transform_constante(img_as_np)

                self.imgs.append(img_as_img)

    def __getitem__(self, index):
        #print("Image load : ",self.files[index])
        single_image_label = self.labels[index]
        img_as_img = self.imgs[index]

        # Transform image to tensor
        if self.transform_tmp is not None:
            img_as_tensor = self.transform_tmp(img_as_img)

        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.files)


INPUT_DATA_DIR = "../cropped/cp/"
IMAGE_SIZE = 128
OUTPUT_DIR = './{date:%Y-%m-%d_%H:%M:%S}/'.format(date=datetime.datetime.now())


if __name__ == "__main__":

    # Name space test
    
    INPUT_DATA_DIR = "../Dataset/FDD/data/kbc_light/"
    
    transformations = transforms.Compose(
        [transforms.Resize(IMAGE_SIZE), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    FDD = FastFDD(INPUT_DATA_DIR, IMAGE_SIZE, IMAGE_SIZE, transformations)

    print(type(FDD), len(FDD))
    
    x = FDD.__getitem__(1)
    print("Vecteur :",x[2])
    print("Label :",x[1])
    #show_tensor(x[0], 1)
    
    
    """
    # Fast Classified Dataset test
    
    INPUT_DATA_DIR = "../Dataset/FDD/data/"
    
    transformations = transforms.Compose(
        [transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    FDD = FastClassifiedDataset(INPUT_DATA_DIR, IMAGE_SIZE, IMAGE_SIZE, transformations)

    print(type(FDD), len(FDD))
    """
    
    """
    # HSV test
    transformations = transforms.Compose(
        [transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    simpsonsDataset = SimpsonsDataset(INPUT_DATA_DIR, IMAGE_SIZE, IMAGE_SIZE, transformations, mode="HSV")

    print(type(simpsonsDataset), len(simpsonsDataset))

    show_tensor(simpsonsDataset.__getitem__(1)[0], 1)"""



    """"
    #DataNoise test
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
     
    #show_tensor(item[0],0)"""

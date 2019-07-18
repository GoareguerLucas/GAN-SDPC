from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision
from torchvision.utils import save_image
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import datetime
import numpy as np
import random
import torch
import time
from itertools import product
import os

import sys
sys.path.append("../")  # ../../GAN-SDPC/

from SimpsonsDataset import *


def weights_init_normal(m, factor=1.0):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        n = float(m.in_channels * m.kernel_size[0] * m.kernel_size[1])
        n += float(m.kernel_size[0] * m.kernel_size[1] * m.out_channels)
        n = n / 2.0
        m.weight.data.normal_(0, np.sqrt(factor / n))
        m.bias.data.zero_()
        #torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        n = float(m.in_features + m.out_features)
        n = n / 2.0
        m.weight.data.normal_(0, np.sqrt(factor / n))
        m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def load_data(path, img_size, batch_size, Fast=True, rand_hflip=False, rand_affine=None, return_dataset=False, mode='RGB'):
    print("Loading data...")
    t_total = time.time()

    # Transformation appliquer avant et pendant l'entraînement
    transform_constante = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), transforms.ToPILImage(mode="RGB")])
    transform_tmp = []
    if rand_hflip:
        transform_tmp.append(transforms.RandomHorizontalFlip(p=0.5))
    if rand_affine != None:
        transform_tmp.append(transforms.RandomAffine(degrees=rand_affine[0], scale=rand_affine[1]))
    transform_tmp = transform_tmp + [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    transform_tmp = transforms.Compose(transform_tmp)
    transform = transforms.Compose([transform_constante, transform_tmp])

    if Fast:
        dataset = FastSimpsonsDataset(path, img_size, img_size, transform_constante, transform_tmp, mode)
    else:
        dataset = SimpsonsDataset(path, img_size, img_size, transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("[Loading Time: ", time.strftime("%Mm:%Ss", time.gmtime(time.time() - t_total)),
          "] [Numbers of samples :", len(dataset), " ]\n")

    if return_dataset == True:
        return dataloader, dataset
    return dataloader


def save_model(model, optimizer, epoch, path):
    print("Save model : ", model._name())
    info = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(info, path)


def load_model(model, optimizer, path):
    print("Load model :", model._name())
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']


def load_models(discriminator, optimizer_D, generator, optimizer_G, n_epochs, model_save_path):
    start_epochD = load_model(discriminator, optimizer_D, model_save_path + "/last_D.pt")
    start_epochG = load_model(generator, optimizer_G, model_save_path + "/last_G.pt")

    if start_epochG is not start_epochD:
        print("Error : G trained different times of D  !!")
        exit(0)
    start_epoch = start_epochD
    if start_epoch >= n_epochs:
        print("Error : Nombre d'epochs demander inférieur au nombre d'epochs déjà effectuer !!")
        exit(0)

    return start_epoch + 1  # La dernière epoch est déjà faite

def sampling(noise, generator, path, epoch, tag=''):
    """
    Utilise generator et noise pour générer une images sauvegarder à path/epoch.png
    Le sample est efféctuer en mode eval pour generator puis il est de nouveau régler en mode train.
    """
    generator.eval()
    gen_imgs = generator(noise)
    save_image(gen_imgs.data[:], "%s/%s%d.png" % (path, tag, epoch), nrow=5, normalize=True)
    generator.train()


def tensorboard_sampling(noise, generator, writer, epoch):
    """
    Sauvegarde des images générer par generator dans writer pour les visualiser avec tensorboard
    """
    generator.eval()
    gen_imgs = generator(noise)
    grid = torchvision.utils.make_grid(gen_imgs, normalize=True)
    writer.add_image('Images générer', grid, epoch)
    generator.train()

def tensorboard_conditional_sampling(noise, label, generator, writer, epoch):
    """
    Sauvegarde des images générer par generator dans writer pour les visualiser avec tensorboard
    Le générateur doit être de type conditionnel et utilise les labels.
    """
    generator.eval()
    gen_imgs = generator(noise, label)
    grid = torchvision.utils.make_grid(gen_imgs, normalize=True)
    writer.add_image('Images générer', grid, epoch)
    generator.train()


def AE_sampling(imgs, encoder, generator, path, epoch):
    generator.eval()
    enc_imgs = encoder(imgs)
    dec_imgs = generator(enc_imgs)
    save_image(imgs.data[:16], "%s/%d_img.png" % (path, epoch), nrow=4, normalize=True)
    save_image(dec_imgs.data[:16], "%s/%d_dec.png" % (path, epoch), nrow=4, normalize=True)
    generator.train()


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    print()


def comp(s):
    s = s.split("/")[-1]  # Nom du fichier
    num = s.split(".")[0]  # Numéro dans le nom du fichier

    return int(num)


def generate_animation(path):
    import imageio
    images_path = glob(path + '[0-9]*.png')

    images_path = sorted(images_path, key=comp)

    images = []
    for i in images_path:
        # print(i)
        images.append(imageio.imread(i))
    imageio.mimsave(path + 'training.gif', images, fps=1)

def scan(exp_name, params, permutation=True, gpu_repart=False):
    """
    Lance le fichier dcgan.py présent dans le dossier courant avec toutes les combinaisons de paramètres possible.
    exp_name : Une chaîne de caractère utiliser pour nommer le sous dossier de résultats tensorboard.
    params : Un dictionnaire où les clefs sont des noms de paramètre (ex : --lrG) et les valeurs sont les différentes
            valeurs à tester pour ce paramètre.
    permutation : Si == True alors toute les permutations (sans répétition) possible de params sont tester,
                  Sinon tout les paramètres sont ziper (tout les paramètres doivent contenir le même nombres d'éléments).
    gpu_repart (Non fonctionnel) : Si plusieurs GPU sont disponible les commandes seront répartis entre eux.
    """
  # Création d'une liste contenant les liste de valeurs à tester
    val_tab = list()
    for v in params.values():
        val_tab.append(v)
        # print(v)
    # print(val_tab)

    # Création d'une liste contenant tout les combinaisons de paramètres à tester
    if permutation:
        perm = list(product(*val_tab))
    else:
        perm = list(zip(*val_tab))
    #print(perm)
    
    # Construction du noms de chaque test en fonction des paramètre qui la compose
    names = list()
    for values in perm:
        b = values
        e = params.keys()
        l = list(zip(e, b))
        l_str = [str(ele) for el in l for ele in el if ele is not "--GPU"]
        names.append(''.join(l_str).replace('-', ''))
    #print(names)

    # Construction de toutes les commandes à lancer
    base = "python3 dcgan.py -r " + exp_name + "/"
    commandes = list()
    for j, values in enumerate(perm):
        com = base + names[j] + "/"
        for i, a in enumerate(params.keys()):
            com = com + " " + str(a) + " " + str(values[i])
        print(com)
        commandes.append(com)
    print("Nombre de commande à lancer :", len(commandes))
    
    # Demande de validation
    print("Valider ? (Y/N)")
    reponse = input()
    #reponse = 'Y'
    if reponse == 'N':
        print("Annulation !")
        exit(0)
    
    """# Répartition sur plusieurs GPU
    if torch.cuda.is_available():
        nb_gpu = torch.cuda.device_count()
        if nb_gpu > 1 and gpu_repart:
            print("Répartition des commandes entre les GPUs : ")
            rep_commandes = list()
            count_gpu = 0
            stock = ""
            for com in commandes:
                stock = stock + com+" --GPU "+str(count_gpu)+" & "
                count_gpu = count_gpu+1
                if count_gpu == nb_gpu:
                    rep_commandes.append(stock[:-3])
                    print(stock[:-3])
                    stock = ""
                    count_gpu = 0
            commandes = rep_commandes
    p = input()"""
    
    # Appelle successif des script avec différents paramètres
    log = list()
    for com in commandes:
        print("Lancement de : ",com)
        ret = os.system(com)
        log.append(ret)
        
    # Récapitulatif
    for idx,com in enumerate(commandes):
        print("Code retour : ",log[idx],"\t| Commandes ", com)


if __name__ == "__main__":

    """D_G_z = np.random.normal(0.5,0.5,100)
    D_x = np.random.normal(0.5,0.5,100)

    plot_scores(D_x,D_G_z)

    print("test")"""

    # generate_animation("W7_128_dcgan/gif/")

    # DataLoader test
    loader, dataset = load_data("../cropped/cp/", 200, 6, Fast=True, rand_hflip=True,
                                rand_affine=[(-25, 25), (1.0, 1.0)], return_dataset=True, mode='RGB')

    for (imgs, _) in loader:
        show_tensor(imgs[1], 1)
        print("Max ", imgs[1].max())
        print("Min ", imgs[1].min())

        exit(0)

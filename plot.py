from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import datetime
import numpy as np
import random
import imageio
import time
import torch

from skimage.color import hsv2rgb

"""
Pour ajouter un plot :
  - Ajouter les stockage des données dans init_hist
  - Ajouter les sauvegarde des données dans save_hist_batch et/ou save_hist_epoch
  - Ajouter une fonction d'affichage propre
  - Appeler la fonction d'affichage dans do_plot
"""


def init_hist(nb_epochs, nb_batch, lossE=False):
    """
    Initialise et retourne un dictionnaire qui servira à sauvegarder les données que l'on voudrais afficher par la suite.
    """

    hist = {}

    # Container for ploting (en minuscule pour les batchs et en majuscule pour les epochs)
    # Losses G et D
    hist["G_losses"] = np.zeros(nb_epochs)
    hist["D_losses"] = np.zeros(nb_epochs)
    hist["g_losses"] = np.zeros(nb_batch)
    hist["d_losses"] = np.zeros(nb_batch)
    if lossE:
        hist["E_losses"] = np.zeros(nb_epochs)
        hist["e_losses"] = np.zeros(nb_batch)
	
    # Moyenne des réponse D(x) et D(G(z)) moyenner par epochs
    hist["D_x_mean"] = np.zeros(nb_epochs)
    hist["D_G_z_mean"] = np.zeros(nb_epochs)
    hist["d_x_mean"] = np.zeros(nb_batch)
    hist["d_g_z_mean"] = np.zeros(nb_batch)

    # Écart type des réponse D(x) et D(G(z)) moyenner par epochs
    hist["D_x_std"] = np.zeros(nb_epochs)
    hist["D_G_z_std"] = np.zeros(nb_epochs)
    hist["d_x_std"] = np.zeros(nb_batch)
    hist["d_g_z_std"] = np.zeros(nb_batch)

    # Coefficient de variation des réponse D(x) et D(G(z)) moyenner par epochs
    hist["D_x_cv"] = np.zeros(nb_epochs)
    hist["D_G_z_cv"] = np.zeros(nb_epochs)
    hist["d_x_cv"] = np.zeros(nb_batch)
    hist["d_g_z_cv"] = np.zeros(nb_batch)

    # Valeur min et max des réponse D(x) et D(G(z)) par epochs
    hist["D_x_min"] = np.ones(nb_epochs)
    hist["D_x_max"] = np.zeros(nb_epochs)
    hist["D_G_z_min"] = np.ones(nb_epochs)
    hist["D_G_z_max"] = np.zeros(nb_epochs)

    return hist


def save_hist_batch(hist, idx_batch, idx_epoch, g_loss, d_loss, d_x, d_g_z, e_loss=-1):
    """
    Sauvegarde les données du batch dans l'historique après traitement
    """

    d_x = d_x.detach().cpu().numpy()
    d_g_z = d_g_z.detach().cpu().numpy()
    g_loss = g_loss.item()
    d_loss = d_loss.item()
    if e_loss != -1:
        e_loss = e_loss.item()
        hist["e_losses"][idx_batch] = e_loss
        
    hist["g_losses"][idx_batch] = g_loss
    hist["d_losses"][idx_batch] = d_loss

    hist["d_x_mean"][idx_batch] = d_x.mean()
    hist["d_g_z_mean"][idx_batch] = d_g_z.mean()

    hist["d_x_std"][idx_batch] = d_x.std()
    hist["d_g_z_std"][idx_batch] = d_g_z.std()

    hist["d_x_cv"][idx_batch] = hist["d_x_std"][idx_batch] / hist["d_x_mean"][idx_batch]
    hist["d_g_z_cv"][idx_batch] = hist["d_g_z_std"][idx_batch] / hist["d_g_z_mean"][idx_batch]

    if d_x.max() > hist["D_x_max"][idx_epoch]:
        hist["D_x_max"][idx_epoch] = d_x.max()
    if d_x.min() < hist["D_x_min"][idx_epoch]:
        hist["D_x_min"][idx_epoch] = d_x.min()
    if d_g_z.max() > hist["D_G_z_max"][idx_epoch]:
        hist["D_G_z_max"][idx_epoch] = d_g_z.max()
    if d_g_z.min() < hist["D_G_z_min"][idx_epoch]:
        hist["D_G_z_min"][idx_epoch] = d_g_z.min()


def save_hist_epoch(hist, idx_epoch, E_losses=False):
    """
    Sauvegarde les données de l'epoch dans l'historique
    """
    
    hist["G_losses"][idx_epoch] = hist["g_losses"].mean()
    hist["D_losses"][idx_epoch] = hist["d_losses"].mean()
    if E_losses:
        hist["E_losses"][idx_epoch] = hist["e_losses"].mean()

    hist["D_x_mean"][idx_epoch] = hist["d_x_mean"].mean()
    hist["D_G_z_mean"][idx_epoch] = hist["d_g_z_mean"].mean()

    hist["D_x_std"][idx_epoch] = hist["d_x_std"].mean()
    hist["D_G_z_std"][idx_epoch] = hist["d_g_z_std"].mean()

    hist["D_x_cv"][idx_epoch] = hist["d_x_cv"].mean()
    hist["D_G_z_cv"][idx_epoch] = hist["d_g_z_cv"].mean()


def do_plot(hist, start_epoch, epoch, E_losses=False):
    # Plot losses
    if E_losses:
        plot_losses(hist["G_losses"], hist["D_losses"], start_epoch, epoch, E_losses=hist["E_losses"])
    else:
        plot_losses(hist["G_losses"], hist["D_losses"], start_epoch, epoch)
    # Plot mean scores
    plot_scores(hist["D_x_mean"], hist["D_G_z_mean"], start_epoch, epoch)
    # Plot std scores
    plot_std_cv(hist["D_x_std"], hist["D_G_z_std"], hist["D_x_cv"], hist["D_G_z_cv"], start_epoch, epoch)
    # Plot min et max scores
    plot_min_max(hist["D_x_max"], hist["D_G_z_max"], hist["D_x_min"], hist["D_G_z_min"], start_epoch, epoch)


def plot_min_max(D_x_max, D_G_z_max, D_x_min, D_G_z_min, start_epoch, current_epoch):
    # Plot game score
    fig = plt.figure(figsize=(10, 5))
    plt.title("Min and max scores for Generator and Discriminator During Training")
    plt.plot(D_x_max, label="Max D(x)")
    plt.plot(D_x_min, label="Min D(x)")
    plt.plot(D_G_z_max, label="Max D(G(z))")
    plt.plot(D_G_z_min, label="Min D(G(z))")
    plt.xlabel("Epochs")
    plt.ylabel("Scores")
    plt.legend()
    # Gradutation
    plt.yticks(np.arange(0.0, 1.2, 0.1))
    positions = np.linspace(0, len(D_x_max), num=6)
    labels = np.linspace(start_epoch - 1, current_epoch, num=6)
    plt.xticks(positions, labels)

    plt.grid(True)
    plt.savefig("min_max.png", format="png")
    plt.close(fig)


def plot_std_cv(D_x_std, D_G_z_std, D_x_cv, D_G_z_cv, start_epoch, current_epoch):
    # Plot game score
    fig = plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator standard deviation and Coefficient of variation on scores During Training")
    plt.plot(D_x_std, label="D(x).std()")
    plt.plot(D_G_z_std, label="D(G(z)).std()")
    plt.plot(D_x_cv, label="CV D(x)")
    plt.plot(D_G_z_cv, label="CV D(G(z))")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    # Gradutation
    plt.yticks(np.arange(0.0, 1.2, 0.1))
    positions = np.linspace(0, len(D_x_std), num=6)
    labels = np.linspace(start_epoch - 1, current_epoch, num=6)
    plt.xticks(positions, labels)

    plt.grid(True)
    plt.savefig("std_cv.png", format="png")
    plt.close(fig)


def plot_scores(D_x, D_G_z, start_epoch=1, current_epoch=-1):
    if len(D_x) <= 0 or len(D_G_z) <= 0:
        return None

    if current_epoch == -1:  # C'est pour surcharger la fonction pour les versions passer
        current_epoch = len(D_x) * 10

    # Plot game score
    fig = plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator scores During Training")
    plt.plot(D_x, label="D(x)")
    plt.plot(D_G_z, label="D(G(z))")
    plt.xlabel("Epochs")
    plt.ylabel("Scores")
    plt.legend()
    # Gradutation
    plt.yticks(np.arange(0.0, 1.2, 0.1))
    positions = np.linspace(0, len(D_x), num=6)
    labels = np.linspace(start_epoch - 1, current_epoch, num=6)
    plt.xticks(positions, labels)

    plt.grid(True)
    plt.savefig("scores.png", format="png")
    plt.close(fig)


def plot_losses(G_losses, D_losses, start_epoch=1, current_epoch=-1, path="losses.png", E_losses=-1):
    if len(G_losses) <= 0 or len(D_losses) <= 0:
        return None

    if current_epoch == -1:  # C'est pour surcharger la fonction pour les versions passer
        current_epoch = len(D_losses) * 10

    # Plot losses
    fig = plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(D_losses, label="D")
    plt.plot(G_losses, label="G")
    if type(E_losses).__module__ == np.__name__:
        plt.plot(E_losses, label="E")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # Gradutation
    positions = np.linspace(0, len(D_losses), num=6)
    labels = np.linspace(start_epoch - 1, current_epoch, num=6)
    plt.xticks(positions, labels)

    plt.grid(True)
    plt.savefig(path, format="png")
    plt.close(fig)


def plot_began(M, k, start_epoch=1, current_epoch=-1):
    if len(M) <= 0 or len(k) <= 0:
        return None

    if current_epoch == -1:  # C'est pour surcharger la fonction pour les versions passer
        current_epoch = len(M) * 10

    # Plot M and k value
    fig = plt.figure(figsize=(10, 5))
    plt.title("M and k Value During Training")
    plt.plot(M, label="M")
    plt.plot(k, label="k")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    # Gradutation
    plt.yticks(np.arange(0.0, 1.2, 0.1))
    positions = np.linspace(0, len(M), num=6)
    labels = np.linspace(start_epoch - 1, current_epoch, num=6)
    plt.xticks(positions, labels)

    plt.grid(True)
    plt.savefig("M_k.png", format="png")
    plt.close(fig)


def plot_lr(lr, start_epoch=1, current_epoch=-1):
    if len(lr) <= 0:
        return None

    if current_epoch == -1:  # C'est pour surcharger la fonction pour les versions passer
        current_epoch = len(lr) * 10

    # Plot lr
    fig = plt.figure(figsize=(10, 5))
    plt.title("lr Value During Training")
    plt.plot(lr, label="lr")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    # Gradutation
    positions = np.linspace(0, len(lr), num=6)
    labels = np.linspace(start_epoch - 1, current_epoch, num=6)
    plt.xticks(positions, labels)

    plt.grid(True)
    plt.savefig("lr.png", format="png")
    plt.close(fig)


def histogram(D_x, D_G_z, epoch, i):
    fig = plt.figure(figsize=(10, 5))
    plt.title("D(x) réponse pour l'epochs " + str(epoch))
    plt.hist(D_x, bins=16)
    plt.scatter(D_x, np.zeros(64), s=60, color='r', marker="|")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.savefig("hist/dx_" + str(epoch) + "_" + str(i) + ".png", format="png")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    plt.title("D(G(z)) réponse pour l'epochs " + str(epoch))
    plt.hist(D_G_z, bins=16)
    plt.scatter(D_G_z, np.zeros(64), s=60, color='r', marker="|")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.savefig("hist/dgz_" + str(epoch) + "_" + str(i) + ".png", format="png")
    plt.close(fig)


def plot_extrem(D_x, D_G_z, nb_batch, start_epoch=1, current_epoch=-1, name="extremum.png"):
    if len(D_x) <= 0 or len(D_G_z) <= 0:
        return None

    # Plot D_x and D_x value
    fig = plt.figure(figsize=(10, 5))
    plt.title("Extrem response of D During Training")
    plt.plot(D_x, label="Log10(D_x.min())")
    plt.plot(D_G_z, label="Log10(D_G_z.min())")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    # Gradutation
    positions = np.linspace(0, current_epoch * nb_batch, num=6)
    labels = np.linspace(start_epoch - 1, current_epoch, num=6)
    plt.xticks(positions, labels)

    plt.grid(True)
    plt.savefig(name, format="png")
    plt.close(fig)


if __name__ == "__main__":

    """D_G_z = np.random.normal(0.5,0.5,100)
    D_x = np.random.normal(0.5,0.5,100)

    plot_scores(D_x,D_G_z)

    print("test")"""

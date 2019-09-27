import sys
import argparse
import numpy as np
from PIL import Image

sys.path.append("../")  # ../../GAN-SDPC/

from utils import *

# ----------
# Script de création d'un gif à partir un dossier contenant des images numéroter dans l'ordre d'apparition. 
# ----------

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--images_path", type=str, default='results/inter1/',
                    help="Chemin vers le dossier contenant les images qui composeront le gif")
parser.add_argument("-f", "--fps", type=int, default=5, help="Frame per seconde")
opt = parser.parse_args()

generate_animation(opt.images_path, fps=opt.fps)

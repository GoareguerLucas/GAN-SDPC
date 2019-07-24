from dcgan import Generator
import sys
import argparse
import numpy as np
from torch.autograd import Variable
import torch

sys.path.append("../")  # ../../GAN-SDPC/

from utils import load_model, sampling

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--model_path", type=str, default='models/last_G.pt',
                    help="Chemin vers le générateur à charger")
parser.add_argument("-s", "--seed_path", type=str, default='seeds.txt',
                    help="Chemin vers le fichier contenant les seeds à générer.")
parser.add_argument("-r", "--results_path", type=str, default='results.png',
                    help="Nom du fichier contenant les résultats")
opt = parser.parse_args()
print(opt)

# Initialize generator 
generator = Generator()
print_network(generator)

# Chargement
load_model(generator, None, opt.model_path)

# Lecture des seeds
with open(opt.seed_path, "r") as f:
    text = f.read().splitlines()
    
    params = []
    for line in text:
        line = line.replace(',','').split()
        print(line)
        params.append(line)
    
    #print(params)
    
# Génération
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
params = Variable(Tensor(np.asarray(params,dtype=np.float64)))
sampling(params, generator, opt.results_path, 0)

import sys

sys.path.append("../")

from utils import *

"""
Script à utiliser pour lancer un modèle avec différent paramètres à la suite, avec gestion des résultats dans tensorboard.
Remplir la variables exp_name avec le nom de l'expérience et le dictionnaire params avec les paramètres à tester pour
keys et leurs valeurs successive pour values.

Pour scanner un nouveau paramètre il suffit d'ajouter l'argument correspondant dans sont dcgan.py. 
Le script utiliser (dcgan.py) doit être compatible avec le paramètre --runs_path (-r) pour utiliser cette fonction 
(cf. W13_current_dcgan/dcgan.py).
Il ne faut pas ajouter au dictionnaire params l'argument --runs_path (-r), tout est gérer dans la fonction scan.
"""

exp_name = "AutoEncoder"

# Suite de scan_params.py, recherche autour de lrD1e05lrE0.0001lrG0.0001

# Dictionnaire des paramètres à tester avec : noms du "paramètre" : [liste des valeurs]
params = {"--lrG": [0.0003, 0.0005], "--lrE": [0.0001], "--lrD": [0.00003, 0.00005], "--GPU": [1]}

scan(exp_name, params, permutation=True)

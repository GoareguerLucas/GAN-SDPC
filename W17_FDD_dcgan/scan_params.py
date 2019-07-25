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

exp_name = "FDDdcgan"

# Dictionnaire des paramètres à tester avec : noms du "paramètre" : [liste des valeurs]
params = {"--lrG": [0.001, 0.0001], "--lrD": [0.0001, 0.00001], "--GPU": [1,1]}

scan(exp_name, params, permutation=False)

import sys

sys.path.append("../")

from utils import *

"""
Script à utiliser pour lancer un modèle avec différent paramètres à la suite.
Remplir la variables exp_name avec le nom de l'expérience et le dictionnaire params avec les paramètres à tester comme
keys et leurs valeurs respective comme values.
"""

exp_name = "Current13"

# Dictionnaire des paramètres à tester avec : noms du "paramètre" : [liste des valeurs]
params = {"-e": [1, 2], "-i": [32, 64]}

scan(exp_name, params)

# GAN-SDPC: Apports du Sparse Coding à un modèle GAN

Les méthodes de génération par réseaux antagonistes (GAN) permettent d'apprendre des modèles génératifs pour de large classes de données d'entrées, notamment pour les images, par exemple des visages. Dans le laboratoire, nous avons développé un paradigme d'apprentissage (Sparse Deep Predictive Coding, SDPC, cf https://laurentperrinet.github.io/publication/boutin-franciosini-ruffier-perrinet-19/) qui permet, via l'utilisation de méthodes de sparse coding, de créer des représentations efficaces pour des images. Comme des croquis, ces modèles utilisent un pattern d'activité le plus parcimonieux possible et sont particulièrement adapté pour modéliser des modèles génératifs. Une conséquence de ces travaux est une meilleure interprétabilité des connections apprises par ces réseaux de neurones. La question qui se pose est de savoir si cette interprétabilité accrue permet aussi d'améliorer la performance des modèles génératifs, comme notamment les GANs.
Dans un premier temps, nous voudrions étudier l'impacte du sparse coding sur les performances ou l'entraînement d'un GAN. Pour cela, nous utiliserons un réseau entrainé sur une classe d'images (visages) comme couche d'entrée du GAN, notamment en utilisant la propriété de reconstruction du réseau sparse.  Dans un second temps et dans l'idée d'améliorer les performances de représentation de notre modèle SDPC nous étudierons les apports éventuels des méthodes de génération par réseaux antagonistes (GAN) pour améliorer la qualité des représentations produites par le réseau SDPC.

Stage M2 dans le cadre du master https://iaaa.lis-lab.fr/


## En cours

Recherche d'un GAN efficace : https://github.com/eriklindernoren/PyTorch-GAN


## Lien utile

Simpsons GAN and dataset : https://github.com/gsurma/image_generator



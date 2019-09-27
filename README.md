# Études des Generative Adversarial Networks

Ce dépôt a pour objectif de stocker le travail réalisé durant le stage final du Master Intelligence Artificielle et Apprentissage Automatique.
Ce stage de six mois s’est déroulé à l’Institut de Neurosciences de La Timone et a été encadré par Laurent Perrinet, chercheur CNRS à l’université d’Aix-Marseille.
Le principal objectif du stage était la compréhension des Generative Adversarial Networks (GAN) appliquée à la génération d’images.
Les GAN permettent d'apprendre des modèles génératifs pour de large classes de données d'entrées, notamment pour les images, par exemple des visages.

Stage M2 dans le cadre du master https://iaaa.lis-lab.fr/

## Structure du dépôt et méthodes de travail

L'essentiel du code produits durant le stage est en python3. 

Le travail a consister en un grand nombres d'expériences et de teste (notamment de proposition d'articles de recherche).  
Chacune des expériences est ranger dans un dossier nommer W*_nom_de_l_experience/ et les comptes rendu des expériences sont grouper par semaines dans les fichiers W*.md.  
Le fichier Makefile contient un ensemble de commande fréquemment utiliser durant les expériences.  
Le fichier requirements.txt peut être utiliser, dans un environnement virtuel de préférence, pour installer les librairies utiliser pour les expériences.  
```
commande : pip install -r requirements.txt  
```

## Dataset utilisés

  ### Simpsons Faces

Nous avons utiliser un dataset fournie par Konstantinos Tokis qui représente des visages de Simpsons. 

Source : https://www.kaggle.com/kostastokis/simpsons-faces

  ### Fractal Dream Dataset

Pour certaines des expériences que nous souhaitions mener il nous fallait un jeu de données dont chaque image puisse être
associées à un point dans l’espace de manière cohérente. Pour ce faire nous avons décidé de construire un
nouveau jeu de données à partir d’un outil mathématique : les Fractal Dream.
Ce dataset a était mis à disposition sur le site Kaggle, ainsi que les codes
utilisés : https://www.kaggle.com/lgoareguer/fractal-dream-dataset#build.py.

Source : http://datashader.org/topics/strange_attractors.html

## Résultats

Certaine expériences et résultats sont disponibles dans les fichiers Rapport_STAGE_M2_IAAA.pdf et Oral de fin de stage.odp.
Vous pouvez également lire les résultats de chacune des expériences mener dans les compte rendue de chaque semaine (cf. W*.md).  

Images générer avec un DCGAN (cf. W17_DCGAN_MAX):

![W17_DCGAN_MAX](readme_images/Results_DCGAN.png "Images générer avec un DCGAN")
 
Images générer avec un DCGAN (cf. W17_AE_MAX): 

![W17_AE_MAX](readme_images/Results_AAE.png "Images générer avec un AAE")

Interpolation dans l'espace latent (cf. W25_Cycle_SF):

![W25_Cycle_SF](readme_images/interpolation.gif "Interpolation dans un DCGAN")

## Ressources

Article présentant une architecture :
  - ![Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
  - ![Adversarial Autoencoder](https://www.cc.gatech.edu/~hays/7476/projects/Avery_Wenchen/)
  - ![Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) 

Implémentation :
  - ![Liste d'un grand nombre d'article sur les GANs](https://github.com/hindupuravinash/the-gan-zoo)
  - ![Nombreux exemples de codes, de courbes et de résultats ](https://github.com/znxlwm/pytorch-generative-model-collections)
  - ![Liste d'application utilisant des GANs](https://github.com/nashory/gans-awesome-applications)
  - ![Implémentations de divers GANs (base de certaines de nos expériences)](https://github.com/eriklindernoren/PyTorch-GAN)

Entraînement des GANs :
  - ![Gan tricks](https://github.com/soumith/ganhacks)
  - ![NIPS 2016 Tutorial:Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf)
  - ![Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)
  
Fonction de coût :
  - ![Analyse théorique](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
  - ![Sens du loss de G](https://github.com/soumith/ganhacks/issues/14)
  - ![Comprendre les losses](https://stackoverflow.com/questions/49420459/what-is-the-ideal-value-of-loss-function-for-a-gan?rq=1)
  - ![Exemples d'évolution de losses](https://stackoverflow.com/questions/42690721/how-to-interpret-the-discriminators-loss-and-the-generators-loss-in-generative)

Bases théoriques et difficultés des GANs :
  - ![Why it is so hard to train Generative Adversarial Networks!](https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b)
  - ![Les GANs d'un point de vus probabilistes](https://medium.com/deep-math-machine-learning-ai/ch-14-general-adversarial-networks-gans-with-math-1318faf46b43)
  
Interpolation :
  - ![SLERP](https://en.wikipedia.org/wiki/Slerp)
  - ![Tutoriel](https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/)

Spécificités techniques :
  - ![Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)
  - ![Détails des deconvolution](https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers)

## Contact

Lucas Goareguer (étudient) : Lucas.Goareguer@etu.univ-amu.fr

Laurent Perrinet (superviseur): Laurent.Perrinet@univ-amu.fr

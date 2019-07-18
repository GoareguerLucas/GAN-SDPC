import numpy as np, pandas as pd, datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import inferno, viridis
from math import sin, cos, sqrt, fabs
from colorcet import palette, fire, kbc
from PIL import Image
import random

# Nombre de points calculer dans chaque images
n= 60000

def trajectory(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=n):
    """
    Calcul des trajectoire des attracteur
    """
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in np.arange(n-1):
        x[i+1], y[i+1] = fn(x[i], y[i], a, b, c, d, e, f)
    return pd.DataFrame(dict(x=x,y=y))

########################################################################
# Diverse fonction de calcul de Strange attractor.
########################################################################

def Fractal_Dream(x, y, a, b, c, d, *o):
    return sin(y*b)+c*sin(x*b), \
           sin(x*a)+d*sin(y*a)

def Hopalong1(x, y, a, b, c, *o):
    return y - sqrt(fabs(b * x - c)) * np.sign(x), \
           a - x

def G(x, mu):
    return mu * x + 2 * (1 - mu) * x**2 / (1.0 + x**2)
def Gumowski_Mira(x, y, a, b, mu, *o):
    xn = y + a*(1 - b*y**2)*y  +  G(x, mu)
    yn = -x + G(xn, mu)
    return xn, yn

def De_Jong(x, y, a, b, c, d, *o):
    return sin(a * y) - cos(b * x), \
           sin(c * x) - cos(d * y)

def Clifford(x, y, a, b, c, d, *o):
    return sin(a * y) + c * cos(a * x), \
           sin(b * x) + d * cos(b * y)

def Symmetric_Icon(x, y, a, b, g, om, l, d, *o):
    zzbar = x*x + y*y
    p = a*zzbar + l
    zreal, zimag = x, y

    for i in range(1, d-1):
        za, zb = zreal * x - zimag * y, zimag * x + zreal * y
        zreal, zimag = za, zb

    zn = x*zreal - y*zimag
    p += b*zn

    return p*x + g*zreal - om*y, \
           p*y - g*zimag + om*x

########################################################################
# Fonction de construction des images
########################################################################

def empty_detection(agg, seuil=0.85):
    """
    Détection des images composer principalement de vide.
    seuil : Définie le niveau de case pleine souhaiter.
    """
    unique, counts = np.unique(agg.values,return_counts=True)
    #print(unique)
    #print(counts)
    
    total = agg.values.size
    if unique[0] == 0:
        nb_zeros = counts[0]
    
    print("Total : ",total) 
    print("Zeros : ",nb_zeros) 
    print("Seuil : ",total*seuil) 
    
    if nb_zeros > total*seuil:
        #print("Trop de pixels vide !")
        return False
    return True

def dsplot(fn, vals, n=n, width = 128, height = 128):
    """Return a PIL image by collecting `n` trajectory points for the given attractor `fn`"""
    df  = trajectory(fn, *vals, n=n)
    cvs = ds.Canvas(plot_width = width, plot_height = height)
    
    agg = cvs.points(df, 'x', 'y')
    
    return agg
    
def rand_vals():
    """
    Retourne des paramètres aléatoire pour générer les datasets.
    """
    #vals = [0.01, 0.01] + list(np.random.random((4))*4-2) +  [random.randint(3,12)]
    #vals = [0, 0] + list(np.random.random((4))*4-2)
    #vals = [0, 0.1] + list(np.random.random((3))*4-2)
    #vals = [0.0, 1.0, 0.008, 0.05] + list(np.random.random((1))*random.uniform(-1,1))  # Gumowski_Mira
    
    #vals = [0.1, 0.1] + list(np.random.random((4))*random.uniform(-3,3)) # Fractal dream
    vals =  list(np.random.random((2))*random.randint(-1,1)) + list(np.random.random((4))*random.uniform(-3,3)) # Fractal dream
    #vals = [0.01, 0.01] + list(np.random.random((5))*random.uniform(-3,3)) + [random.randint(3,9)] # Symetric Icon
    
    return vals
    
def build_dataset(size=5, seuil=0.85):
    """
    Construit un dataset de size images en excluant les images trop vide (seuil).
    """
    succes = False
    count = 0
    
    cmaps = [inferno,viridis,kbc]
    name_cmaps = ["inferno","viridis","kbc"]
    
    while count < size:
        while not succes:
            # Génération de paramètres
            vals= rand_vals()
            print("Params : ",vals)
            
            # Calcul de l'image
            agg = dsplot(Fractal_Dream, vals=vals)
            
            # Vérification de l'image
            succes = empty_detection(agg, seuil=seuil)
        
        # Conversion en PIL image
        img = tf.shade(agg, cmap=cmaps[count%3], alpha=255)
        img = tf.Image.to_pil(img)
        
        # Suppression de la couche Alpha
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        
        
        name = ("{}, "*(len(vals)-1)+" {}").format(*vals).replace(", ","_")
        background.save("data/"+name_cmaps[count%3] +'/'+ str(name)+'.png', 'PNG', quality=100)
        
        count = count+1
        succes = False
        
    return count
        
def one_image(fonction=Symmetric_Icon, vals=[0.01, 0.01, 1.8, 0.0, 1.0, 0.1, -1.93, 5], color=inferno, path='test.png', alpha=True, width=128, height=128):
    """
    Construit et sauvegarde une image.
    Paramètres :
        fonction : La fonction à utiliser pour calculer l'attracteur.
        vals : Les paramètres à fournir à la fonction pour calculer l'attracteur
        color : La palette de couleur à utiliser.
        path : Le chemin où sauvegarder l'image.
        alpha : Si True alors le fond de l'image sera transparent sinon il sera blanc.  
        width et height : largeur et longueur de l'image. 
    """
    agg = dsplot(fonction, vals=vals, width=width, height=height)

    # Conversion en PIL image
    img = tf.shade(agg, cmap=color, alpha=255)
    img = tf.Image.to_pil(img)

    # Suppression de la couche Alpha
    if not alpha:
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        background.save(path, 'PNG', quality=100)
    else: # Ou non
        img.save(path, 'PNG', quality=100)
        
palette["viridis"]=viridis
palette["inferno"]=inferno
palette["kbc"]=kbc

#build_dataset(size=12000)

one_image(path='test2.png')
one_image(path='test3.png',alpha=False)

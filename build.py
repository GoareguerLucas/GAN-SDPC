import numpy as np, pandas as pd, datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import inferno, viridis
from math import sin, cos, sqrt, fabs
from colorcet import palette, fire, kbc
from PIL import Image
import random

n=200000

def trajectory(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=n):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in np.arange(n-1):
        x[i+1], y[i+1] = fn(x[i], y[i], a, b, c, d, e, f)
    return pd.DataFrame(dict(x=x,y=y))

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

def empty_detection(agg, seuil=0.6):
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

def dsplot(fn, vals, n=n):
    """Return a PIL image by collecting `n` trajectory points for the given attractor `fn`"""
    df  = trajectory(fn, *vals, n=n)
    cvs = ds.Canvas(plot_width = 128, plot_height = 128)
    
    agg = cvs.points(df, 'x', 'y')
    
    return agg
#random.uniform(-3,3), random.uniform(-5,5), random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,3),
def rand_vals():
	vals = [0.01, 0.01] + list(np.random.random((4))*4-2) +  [random.randint(3,12)]
	#vals = list(np.round_(vals[:-1],3)) + [random.randint(3,7)]
	return vals
	
def build_dataset(size=5):
	succes = False
	count = 0
	
	cmaps = [inferno,viridis,kbc]
	name_cmaps = ["inferno","viridis","kbc"]
	
	for i in range(size):
		while not succes:
			# Génération de paramètres
			vals= rand_vals()
			#print("Params : ",vals)
			
			# Calcul de l'image
			agg = dsplot(Symmetric_Icon, vals=vals)
			
			# Vérification de l'image
			succes = empty_detection(agg)
		
		# Conversion en PIL image
		img = tf.shade(agg, cmap=cmaps[count%3], alpha=255)
		img = tf.Image.to_pil(img)
		
		# Suppression de la couche Alpha
		background = Image.new("RGB", img.size, (255, 255, 255))
		background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
		
		
		name = ("{}, "*(len(vals)-1)+" {}").format(*vals).replace(", ","_")
		background.save(name_cmaps[count%3] + str(name)+'.png', 'PNG', quality=100)
		
		count = count+1
		succes = False
		
	return count
		
palette["viridis"]=viridis
palette["inferno"]=inferno
palette["kbc"]=kbc

build_dataset(size=12000)

""""agg = dsplot(Symmetric_Icon, vals=[0.01, 0.01, 1.8, 0.0, 1.0, 0.1, -1.93, 5])
succes = empty_detection(agg)

# Conversion en PIL image
img = tf.shade(agg, cmap=kbc, alpha=255)
img = tf.Image.to_pil(img)

# Suppression de la couche Alpha
background = Image.new("RGB", img.size, (255, 255, 255))
background.paste(img, mask=img.split()[3]) # 3 is the alpha channel

background.save('test.png', 'PNG', quality=100)"""

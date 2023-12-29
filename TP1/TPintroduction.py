#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% SECTION 1 inclusion de packages externes 


import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image 
from skimage import io as skio


#%% SECTION 2 fonctions utiles pour le TP

def viewimage(im,normalise=True,MINI=0.0, MAXI=255.0):
    """ Cette fonction fait afficher l'image EN NIVEAUX DE GRIS 
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee 
        entre 0 et 255 avant d'être sauvegardee.
        Si normalise=False MINI et MAXI seront mis a 0 et 255 dans l'image resultat
        
    """
    imt=np.float32(im.copy())
    if platform.system()=='Darwin': #on est sous mac
        prephrase='open -a GIMP '
        endphrase=' ' 
    elif platform.system()=='Windows': 
        #ou windows ; probleme : il faut fermer gimp pour reprendre la main; 
        #si vous savez comment faire (commande start ?) je suis preneur 
        prephrase='"C:/Program Files/GIMP 2/bin/gimp-2.10.exe" '
        endphrase=' '
    else: #SINON ON SUPPOSE LINUX (si vous avez un windows je ne sais pas comment faire. Si vous savez dites-moi.)
        prephrase='gimp '
        endphrase= ' &'
    
    if normalise:
        m=im.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=255*imt/M

    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
        imt *= 255
    
    nomfichier=tempfile.mktemp('TPIMA.png')
    commande=prephrase +nomfichier+endphrase
    imt = imt.astype(np.uint8)
    skio.imsave(nomfichier,imt)
    os.system(commande)

def viewimage_color(im,normalise=True,MINI=0.0, MAXI=255.0):
    """ Cette fonction fait afficher l'image EN NIVEAUX DE GRIS 
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee 
        entre 0 et 255 avant d'être sauvegardee.
        Si normalise=False MINI(defaut 0) et MAXI (defaut 255) seront mis a 0 et 255 dans l'image resultat
        
    """
    imt=np.float32(im.copy())
    if platform.system()=='Darwin': #on est sous mac
        prephrase='open -a GIMP '
        endphrase= ' '
    elif platform.system()=='Windows': 
        #ou windows ; probleme : il faut fermer gimp pour reprendre la main; 
        #si vous savez comment faire (commande start ?) je suis preneur 
        prephrase='"C:/Program Files/GIMP 2/bin/gimp-2.10.exe" '
        endphrase=' '
    else: #SINON ON SUPPOSE LINUX (si vous avez un windows je ne sais comment faire. Si vous savez dites-moi.)
        prephrase='gimp '
        endphrase=' &'
    
    if normalise:
        m=imt.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=255*imt/M
    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
        imt *= 255
    
    nomfichier=tempfile.mktemp('TPIMA.pgm')
    commande=prephrase +nomfichier+endphrase
    imt = imt.astype(np.uint8)
    skio.imsave(nomfichier,imt)
    os.system(commande)

def noise(im,br):
    """ Cette fonction ajoute un bruit blanc gaussien d'ecart type br
       a l'image im et renvoie le resultat"""
    imt=np.float32(im.copy())
    sh=imt.shape
    bruit=br*np.random.randn(*sh)
    imt=imt+bruit
    return imt

def quantize(im,n=2):
    """
    Renvoie une version quantifiee de l'image sur n (=2 par defaut) niveaux  
    """
    imt=np.float32(im.copy())
    if np.floor(n)!= n or n<2:
        raise Exception("La valeur de n n'est pas bonne dans quantize")
    else:
        m=imt.min()
        M=imt.max()
        imt=np.floor(n*((imt-m)/(M-m)))*(M-m)/n+m
        imt[imt==M]=M-(M-m)/n #cas des valeurs maximales
        return imt
    

def seuil(im,s):
    """ renvoie une image blanche(255) la ou im>=s et noire (0) ailleurs.
    """
    imt=np.float32(im.copy())
    mask=imt<s
    imt[mask]=0
    imt[~mask]=255
    return imt

def gradx(im):
    "renvoie le gradient dans la direction x"
    imt=np.float32(im)
    gx=0*imt
    gx[:,:-1]=imt[:,1:]-imt[:,:-1]
    return gx

def grady(im):
    "renvoie le gradient dans la direction y"
    imt=np.float32(im)
    gy=0*imt
    gy[:-1,:]=imt[1:,:]-imt[:-1,:]
    return gy

def view_spectre(im,option=1,hamming=False):
    """ affiche le spectre d'une image
     si option =1 on affiche l'intensite de maniere lineaire
     si option =2 on affiche le log
     si hamming=True (defaut False) alors une fenetre de hamming est appliquee avant de prendre la transformee de Fourier
     """
    imt=np.float32(im.copy())
    (ty,tx)=im.shape
    pi=np.pi
    if hamming:
        XX=np.ones((ty,1))@(np.arange(0,tx).reshape((1,tx)))
        YY=(np.arange(0,ty).reshape((ty,1)))@np.ones((1,tx))
        imt=(1-np.cos(2*pi*XX/(tx-1)))*(1-np.cos(2*pi*YY/(ty-1)))*imt
    aft=np.fft.fftshift(abs(np.fft.fft2(imt)))
    
    if option==1:
        viewimage(aft)
    else:
        viewimage(np.log(0.1+aft))


def filterlow(im): 
    """applique un filtre passe-bas parfait a une image (taille paire)"""
    (ty,tx)=im.shape
    imt=np.float32(im.copy())
    pi=np.pi
    XX=np.concatenate((np.arange(0,tx/2+1),np.arange(-tx/2+1,0)))
    XX=np.ones((ty,1))@(XX.reshape((1,tx)))
    
    YY=np.concatenate((np.arange(0,ty/2+1),np.arange(-ty/2+1,0)))
    YY=(YY.reshape((ty,1)))@np.ones((1,tx))
    mask=(abs(XX)<tx/4) & (abs(YY)<ty/4)
    imtf=np.fft.fft2(imt)
    imtf[~mask]=0
    return np.real(np.fft.ifft2(imtf))

def filtergauss(im):
    """applique un filtre passe-bas gaussien. coupe approximativement a f0/4"""
    (ty,tx)=im.shape
    imt=np.float32(im.copy())
    pi=np.pi
    XX=np.concatenate((np.arange(0,tx/2+1),np.arange(-tx/2+1,0)))
    XX=np.ones((ty,1))@(XX.reshape((1,tx)))
    
    YY=np.concatenate((np.arange(0,ty/2+1),np.arange(-ty/2+1,0)))
    YY=(YY.reshape((ty,1)))@np.ones((1,tx))
    # C'est une gaussienne, dont la moyenne est choisie de sorte que
    # l'integrale soit la meme que celle du filtre passe bas
    # (2*pi*sig^2=1/4*x*y (on a suppose que tx=ty))
    sig=(tx*ty)**0.5/2/(pi**0.5)
    mask=np.exp(-(XX**2+YY**2)/2/sig**2)
    imtf=np.fft.fft2(imt)*mask
    return np.real(np.fft.ifft2(imtf))

#%% SECTION 3 exemples de commandes pour effectuer ce qui est demande pendant le TP


# Partie Changement de contraste


#%% charger une image

im=skio.imread('images/maison.tif')

# connaitre la taille de l'image
shape = im.shape
print(shape)

#avoir la valeur d'un pixel
pixel_value = im[9,8] #pixel en y=9 et x=8
print(pixel_value)

# visualiser l'image (elle en niveaux de gris)
#viewimage(im)

# afficher une ligne d'une image comme un signal 
# par exemple on affiche la ligne y=129


#plt.plot(im[129,:])
# colonne x=45
#plt.plot(im[:,45])

# maximum d'une image
im.max()

# minimum
im.min()

#minimum d'une ligne
im[129,:].min()

# transformation en image a valeurs reelles
imfloat= np.float32(im)
# valeur absolue d'une image
abs(im)

# pour obtenir de l'aide 
help(quantize)

# transformer un tableau en une line
r=im.reshape( ( -1,))
print (r.shape)

#%% une image couleur 
im=skio.imread('images/fleur.tif')
#viewimage_color(im,normalise=False) #on ne normalise pas pour garder l'image comme
                                    # a l'origine
# voir un seul canal (rouge)
#viewimage(im[:,:,0])
#viewimage(im.mean(axis=2)) #la moyenne des trois canaux
#%%
#histogrammes
# simple visualisation
im=skio.imread('images/maison.tif')
#plt.hist(im.reshape((-1,)),bins=255) #le reshape transforme en tableau 1D
#%%
#calcul d'un histogramme cumule
(histo,bins)=np.histogram(im.reshape((-1,)),np.arange(0,256)) #le reshape est inutile pour np.histogram, mais on le laisse pour la compatibilite avec plt.hist
histo=histo/histo.sum()
histocum=histo.cumsum()
#plt.plot(histocum)

#%% ajout de bruit 
imbr=noise(im,10)
#viewimage_color(imbr,normalise=False)
#effet sur l'histogramme
#plt.hist(im.reshape((-1,)),255)
#plt.show()
#plt.hist(imbr.reshape((-1,)),255)
#plt.show()


#%% egalisation d'histogramme
im=skio.imread('images/sombre.jpg')
im=im.mean(axis=2) #on est sur que l'image est grise
#viewimage(im)
(histo,bins)=np.histogram(im.reshape((-1,)),np.arange(0,256)) #le reshape en inutile pour np.histogram, mais on le laisse pour la compatibilite avec plt.hist
histo=histo/histo.sum()
histocum=histo.cumsum()
plt.plot(histocum)
plt.show()


imequal=histocum[np.uint8(im)]*255
(histo2,bins2) = np.histogram(imequal.reshape((-1,)),np.arange(0,256))
histo2 = histo2/histo2.sum()
histocum2 = histo2.cumsum()
plt.plot(histocum2)
plt.show()


#viewimage(imequal)

#%% 
u=skio.imread('images/vue1.tif')
v=skio.imread('images/vue2.tif')

plt.hist(u.reshape((-1,)),bins=255)
#plt.show()
plt.hist(v.reshape((-1,)),bins=255)
#plt.show()




#viewimage(u)
#viewimage(v)
# TEXTE1 dans le texte du tp
ind=np.unravel_index(np.argsort(u, axis=None), u.shape) #unravel donne les index 2D a partir des index 1d renvoyes par argsort (axis=None)
unew=np.zeros(u.shape,u.dtype)
unew[ind]=np.sort(v,axis=None)

#viewimage(abs(np.float32(unew)-np.float32(v)))


#viewimage(unew) #u avec l'histogramme de v

#DE MANIERE EQUIVALENTE et Peut-etre plus claire

ushape=u.shape
uligne=u.reshape((-1,)) #transforme en ligne
vligne=v.reshape((-1,))
ind=np.argsort(uligne)
unew=np.zeros(uligne.shape,uligne.dtype)
unew[ind]=np.sort(vligne)
# on remet a la bonne taille
unew=unew.reshape(ushape)
#viewimage(unew)


#viewimage(abs(np.float32(u)-np.float32(v)))
#viewimage(abs(np.float32(unew)-np.float32(u)))


#%% quantification dithering 

im=skio.imread('images/maison.tif')
im2=quantize(im,2)
viewimage(im2)

noisy = noise(im,10)
noisy2 = quantize(noisy, 2)
#viewimage(noisy2)

#viewimage(seuil(noise(im,40),128)) #exemple de dithering

#%% log d'un histogramme
#plt.plot(np.log(np.histogram(gradx(im),255)[0]))



#%%
im=skio.imread('images/facade.tif')
view_spectre(im,option=2,hamming=False)
#%% FIN TP INTRODUCTION

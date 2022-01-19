
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 21:23:18 2022

@author: Florian
"""

import matplotlib.pyplot as plt	# for plots
from PIL import Image
import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import imageio
from os import listdir
from os.path import isfile, join

def importimage(pfad):
    """
    Parameters
    ----------
    pfad : Bildpfad für Bild welches eingelesen werden soll und geplottet wird
    Returns
    -------
    Bild als 2-D Numpy-Arrays
    """
    img = imageio.imread(pfad)    # Bild als schwarz/weiß Tiff-Datei einlesen
    return img

def getdarkorwhiteimg(path):
    img_names = [f for f in listdir(path) if isfile(join(path, f))]
    images = []
    for name in img_names:
        images.append(importimage(path + name))
    
    img_mean = np.zeros(images[0].shape)
    for i in range(len(images)) : 
        img_mean = img_mean + images[i]
    img_mean = img_mean/len(images)
    return img_mean
    
def interpolat_dark_image(y_dark_t1, y_dark_t2, t1, t2, texp):
    y_dark_texp = (t2-texp)/(t2-t1)*y_dark_t1 + (texp-t1)/(t2-t1)*y_dark_t2
    return y_dark_texp

'Dunkel- und Hellbilder importieren'
y_dark_t4500 = getdarkorwhiteimg('Test_Aufgabe_5_neu/Dunkelbild/B_50/')
y_dark_t1000 = getdarkorwhiteimg('Test_Aufgabe_5_neu/Dunkelbild/B_5000/')
y_50 = getdarkorwhiteimg('Test_Aufgabe_5_neu/Optik/Neu/')

'Dunkelbild interpolieren'
y_dark_texp = interpolat_dark_image(y_dark_t4500, y_dark_t1000, 50, 5000, 200000)
'Weißbild korrigieren'
y_50i = y_50 - y_dark_texp

'Zu korrigierendes Bild importieren'
y = importimage('Test_Aufgabe_5_neu/Testchart/LP_25_B_4_A_60.tiff')

yi = np.mean(y_50i)*(y - y_dark_texp)/y_50i

plt.imshow(y, cmap='gray')
plt.show()
plt.imshow(yi, cmap='gray')
plt.show()



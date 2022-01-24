
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:03:32 2022

@author: Milan Kaiser
"""
## MIT BILDERN U-KUGEL

import matplotlib.pyplot as plt	# for plots
from PIL import Image
import numpy as np
from scipy import stats
import scipy as sp
from scipy.ndimage.filters import gaussian_filter
import imageio
import os
from os import listdir
from os.path import isfile, join
from scipy.optimize import curve_fit

def importimage(pfad):
    """w
    Parameters
    ----------
    pfad : Bildpfad für Bild welches eingelesen werden soll und geplottet wird
    Returns
    -------
    Bild als 2-D Numpy-Arrays
    """
    img = imageio.imread(pfad)    # Bild als schwarz/weiß Tiff-Datei einlesen
    return img

def mittelgrau(img1, img2):
    'Formel Vorl. 3 Folie 13/23'
    x = 1/(2*img1.shape[0]*img1.shape[1]) * np.sum(img1)+1/(2*img1.shape[0]*img1.shape[1] )*np.sum(img2)
    return x

def zeitrauschen(img1, img2):
    'Formel Vorl. 3 Folie 13/23'
    x = 1/(2*img1.shape[0]*img1.shape[1]) * np.sum((img1-img2)**2)
    return x

# Define the Gaussian function
def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y

## Aufgabe 1
'Parameter eingeben'
A = 1.67e-6*1.67e-6               # Pixelfläche [m^2]
wellenlaenge = 475e-9   # Wellenlänge [m]
t = np.array([1,2,3,4,5,6,7,8,9,10]) # Belichtungszeit [ms]
t_exp = t*1e-3

Kalibrierfaktor =3.352e-9 # in A/lx
Beleuchtungsstaerke = 7.82e-6 / Kalibrierfaktor     # in lx
kmax_skoptisch = 1699           # lm/W
v_skoptisch = 0.24 #0.58
E = Beleuchtungsstaerke / kmax_skoptisch / v_skoptisch
print("Beleuchtungsstärke: " + str(E))
# E = 0.1                      # Bestrahlungsstärke [W/m^2]

u_p = 5.034e24 * A * E * t_exp * wellenlaenge

'Bilder importieren'
basepath=r"..\neu\hell"
A_40_B_1000_1 = importimage(os.path.join(basepath, '1_1.bmp'))
A_40_B_1000_2 = importimage(os.path.join(basepath, '1_2.bmp'))
A_40_B_2000_1 = importimage(os.path.join(basepath, '2_1.bmp'))
A_40_B_2000_2 = importimage(os.path.join(basepath, '2_2.bmp'))
A_40_B_3000_1 = importimage(os.path.join(basepath, '3_1.bmp'))
A_40_B_3000_2 = importimage(os.path.join(basepath, '3_2.bmp'))
A_40_B_4000_1 = importimage(os.path.join(basepath, '4_1.bmp'))
A_40_B_4000_2 = importimage(os.path.join(basepath, '4_2.bmp'))
A_40_B_5000_1 = importimage(os.path.join(basepath, '5_1.bmp'))
A_40_B_5000_2 = importimage(os.path.join(basepath, '5_2.bmp'))
A_40_B_6000_1 = importimage(os.path.join(basepath, '6_1.bmp'))
A_40_B_6000_2 = importimage(os.path.join(basepath, '6_2.bmp'))
A_40_B_7000_1 = importimage(os.path.join(basepath, '7_1.bmp'))
A_40_B_7000_2 = importimage(os.path.join(basepath, '7_2.bmp'))
A_40_B_8000_1 = importimage(os.path.join(basepath, '8_1.bmp'))
A_40_B_8000_2 = importimage(os.path.join(basepath, '8_2.bmp'))
A_40_B_9000_1 = importimage(os.path.join(basepath, '9_1.bmp'))
A_40_B_9000_2 = importimage(os.path.join(basepath, '9_2.bmp'))
A_40_B_10000_1 = importimage(os.path.join(basepath, '10_1.bmp'))
A_40_B_10000_2 = importimage(os.path.join(basepath, '10_2.bmp'))

basepath=r"..\neu\dunkel"

A_40_B_1000_ABG_1 = importimage(os.path.join(basepath, '1_1.bmp'))
A_40_B_1000_ABG_2 = importimage(os.path.join(basepath, '1_2.bmp'))
A_40_B_2000_ABG_1 = importimage(os.path.join(basepath, '2_1.bmp'))
A_40_B_2000_ABG_2 = importimage(os.path.join(basepath, '2_2.bmp'))
A_40_B_3000_ABG_1 = importimage(os.path.join(basepath, '3_1.bmp'))
A_40_B_3000_ABG_2 = importimage(os.path.join(basepath, '3_2.bmp'))
A_40_B_4000_ABG_1 = importimage(os.path.join(basepath, '4_1.bmp'))
A_40_B_4000_ABG_2 = importimage(os.path.join(basepath, '4_2.bmp'))
A_40_B_5000_ABG_1 = importimage(os.path.join(basepath, '5_1.bmp'))
A_40_B_5000_ABG_2 = importimage(os.path.join(basepath, '5_2.bmp'))
A_40_B_6000_ABG_1 = importimage(os.path.join(basepath, '6_1.bmp'))
A_40_B_6000_ABG_2 = importimage(os.path.join(basepath, '6_2.bmp'))
A_40_B_7000_ABG_1 = importimage(os.path.join(basepath, '7_1.bmp'))
A_40_B_7000_ABG_2 = importimage(os.path.join(basepath, '7_2.bmp'))
A_40_B_8000_ABG_1 = importimage(os.path.join(basepath, '8_1.bmp'))
A_40_B_8000_ABG_2 = importimage(os.path.join(basepath, '8_2.bmp'))
A_40_B_9000_ABG_1 = importimage(os.path.join(basepath, '9_1.bmp'))
A_40_B_9000_ABG_2 = importimage(os.path.join(basepath, '9_2.bmp'))
A_40_B_10000_ABG_1 = importimage(os.path.join(basepath, '10_1.bmp'))
A_40_B_10000_ABG_2 = importimage(os.path.join(basepath, '10_2.bmp'))


'Mittlerer Grauwert'
'Helldbilder'
u_y = np.zeros(10)
u_y[0] = mittelgrau(A_40_B_1000_1,A_40_B_1000_2)
u_y[1] = mittelgrau(A_40_B_2000_1,A_40_B_2000_2)
u_y[2] = mittelgrau(A_40_B_3000_1,A_40_B_3000_2)
u_y[3] = mittelgrau(A_40_B_4000_1,A_40_B_4000_2)
u_y[4] = mittelgrau(A_40_B_5000_1,A_40_B_5000_2)
u_y[5] = mittelgrau(A_40_B_6000_1,A_40_B_6000_2)
u_y[6] = mittelgrau(A_40_B_7000_1,A_40_B_7000_2)
u_y[7] = mittelgrau(A_40_B_8000_1,A_40_B_8000_2)
u_y[8] = mittelgrau(A_40_B_9000_1,A_40_B_9000_2)
u_y[9] = mittelgrau(A_40_B_10000_1,A_40_B_10000_2)

'Dunkelbilder'
u_y_dark = np.zeros(10)
u_y_dark[0] = mittelgrau(A_40_B_1000_ABG_1,A_40_B_1000_ABG_2)
u_y_dark[1] = mittelgrau(A_40_B_2000_ABG_1,A_40_B_2000_ABG_2)
u_y_dark[2] = mittelgrau(A_40_B_3000_ABG_1,A_40_B_3000_ABG_2)
u_y_dark[3] = mittelgrau(A_40_B_4000_ABG_1,A_40_B_4000_ABG_2)
u_y_dark[4] = mittelgrau(A_40_B_5000_ABG_1,A_40_B_5000_ABG_2)
u_y_dark[5] = mittelgrau(A_40_B_6000_ABG_1,A_40_B_6000_ABG_2)
u_y_dark[6] = mittelgrau(A_40_B_7000_ABG_1,A_40_B_7000_ABG_2)
u_y_dark[7] = mittelgrau(A_40_B_8000_ABG_1,A_40_B_8000_ABG_2)
u_y_dark[8] = mittelgrau(A_40_B_9000_ABG_1,A_40_B_9000_ABG_2)
u_y_dark[9] = mittelgrau(A_40_B_10000_ABG_1,A_40_B_10000_ABG_2)

'Zeitliches Rauschen'
'Hellbilder'
s_y = np.zeros(10)
s_y[0] = zeitrauschen(A_40_B_1000_1,A_40_B_1000_2)
s_y[1] = zeitrauschen(A_40_B_2000_1,A_40_B_2000_2)
s_y[2] = zeitrauschen(A_40_B_3000_1,A_40_B_3000_2)
s_y[3] = zeitrauschen(A_40_B_4000_1,A_40_B_4000_2)
s_y[4] = zeitrauschen(A_40_B_5000_1,A_40_B_5000_2)
s_y[5] = zeitrauschen(A_40_B_6000_1,A_40_B_6000_2)
s_y[6] = zeitrauschen(A_40_B_7000_1,A_40_B_7000_2)
s_y[7] = zeitrauschen(A_40_B_8000_1,A_40_B_8000_2)
s_y[8] = zeitrauschen(A_40_B_9000_1,A_40_B_9000_2)
s_y[9] = zeitrauschen(A_40_B_10000_1,A_40_B_10000_2)

'Dunkelbilder'
s_y_dark = np.zeros(10)
s_y_dark[0] = zeitrauschen(A_40_B_1000_ABG_1,A_40_B_1000_ABG_2)
s_y_dark[1] = zeitrauschen(A_40_B_2000_ABG_1,A_40_B_2000_ABG_2)
s_y_dark[2] = zeitrauschen(A_40_B_3000_ABG_1,A_40_B_3000_ABG_2)
s_y_dark[3] = zeitrauschen(A_40_B_4000_ABG_1,A_40_B_4000_ABG_2)
s_y_dark[4] = zeitrauschen(A_40_B_5000_ABG_1,A_40_B_5000_ABG_2)
s_y_dark[5] = zeitrauschen(A_40_B_6000_ABG_1,A_40_B_6000_ABG_2)
s_y_dark[6] = zeitrauschen(A_40_B_7000_ABG_1,A_40_B_7000_ABG_2)
s_y_dark[7] = zeitrauschen(A_40_B_8000_ABG_1,A_40_B_8000_ABG_2)
s_y_dark[8] = zeitrauschen(A_40_B_9000_ABG_1,A_40_B_9000_ABG_2)
s_y_dark[9] = zeitrauschen(A_40_B_10000_ABG_1,A_40_B_10000_ABG_2)

'Y-Achse Sensitivität/Photonentransver (Vorl. 3 Folie 14/23'
delta_u = u_y-u_y_dark
delta_s = s_y-s_y_dark

'Sensitivitätskurve Plotten'
plt.plot(u_p, delta_u,'o', label='Data')
x_sens = np.linspace(1,400000,10)
res_sens = stats.linregress(u_p[0:7], delta_u[0:7])
plt.plot(x_sens,res_sens.intercept + res_sens.slope*x_sens, label='fit')
plt.title('Sensitivität (U-Kugel)')
plt.xlabel('irradiation in photons/pixel')
plt.ylabel('gray value - dark value $µ_{y}−µ_{y_{dark}}$')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.legend()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid()
plt.show()

'Photonentransferkurve Plotten'
plt.plot(delta_u,delta_s,'o', label='Data')
x_photo = np.linspace(1,230,10)
res_sens = stats.linregress(delta_u[:7],delta_s[:7])
plt.plot(x_photo,res_sens.intercept + res_sens.slope*x_photo, label='fit')
plt.title('Photonentransfer (U-Kugel)')
plt.xlabel('gray value - dark value $µ_y−µ_{y_{dark}}$')
plt.ylabel('variance gray value $σ_y^2−σ^2_{y_{dark}}$')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.legend()
plt.grid()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.show()

'Kennwerte Bestimmen'
'Bei K und R nur 70% der Werte nutzen'
R = stats.linregress(u_p[:7], delta_u[:7])[0]
K = stats.linregress(delta_u[:7], delta_s[:7])[0]
eta = R/K
s_q = 1/12
#s_d = np.abs((s_y_dark[0]-s_q))/K**2
s_d= np.abs((s_y_dark[0]-s_q)/K**2)
'Kennwerte ausgeben'
print('R = ', + R)
print('K = ', + K)
print('eta = ', eta*100,'%')
print('σ^2_d = ', + s_d)

## Aufgabe 2
# 'SNR-Kurve'
SNR = (eta * u_p)/np.sqrt(s_d+s_q/K**2 + eta * u_p)
# 'Data plotten:'
plt.loglog(u_p,SNR,'o', label='Data')
x_snr = np.linspace(1,np.max(u_p),1000)
# 'theor. limit plotten:'
plt.loglog(x_snr,np.sqrt(x_snr), label='theor. limit')
plt.title('SNR-Kurve (U-Kugel)')
plt.xlabel('irradiation in photons/pixel')
plt.ylabel('SNR')
plt.grid(True, which="both", ls="-")
plt.xlim(xmin=1)
plt.ylim(ymin=0.1)

'u_p min und u_p sat, senkrechte Linien plotten'
plt.vlines(x=u_p[8], ymin=0, ymax=10000, color='black', linestyles='--')
plt.text(u_p[8],0.5,'Saturation',rotation=90)
u_pmin = 1/eta * (np.sqrt(s_d+s_q/K**2)+0.5)
plt.vlines(x=u_pmin, ymin=0, ymax=10000, color='black', linestyles='--')
plt.text(u_pmin,0.5,'threshold',rotation=90)
DR = 20 * np.log10(u_p[8]/u_pmin)

print("DR = ",  DR, " dB")

'Fit plotten'
snr_fit = (eta * x_snr)/np.sqrt(s_d+s_q/K**2 + eta * x_snr)
plt.loglog(x_snr, snr_fit, label='fit')
plt.legend()
plt.show()

## Aufgabe 3
'Importieren der Bilder'
path_img_bright = r"..\neu\raeumliche_inhomo\hell"
path_img_dark = r"..\neu\raeumliche_inhomo\dunkel"
imgnames_bright = [f for f in listdir(path_img_bright) if isfile(os.path.join(path_img_bright, f))]
imgnames_dark = [f for f in listdir(path_img_dark) if isfile(os.path.join(path_img_dark, f))]

A3_bright = []
A3_dark = []

for name in imgnames_bright:
     A3_bright.append(importimage(os.path.join(path_img_bright, name)))
for name in imgnames_dark:
     A3_dark.append(importimage(os.path.join(path_img_dark, name)))

'Dunkelbild'
y_dark = np.zeros(A3_dark[0].shape)
for i in range(16) : 
    y_dark = y_dark + A3_dark[i]
y_dark = y_dark/16

plt.imshow(y_dark,cmap='gray')
plt.title("Dunkelbild")
plt.show()

'Hellbild'
y_50 = np.zeros(A3_bright[0].shape)
for i in range(16) : 
    y_50 = y_50 + A3_bright[i]
y_50 = y_50/16



plt.imshow(y_50,cmap='gray')
plt.title("Hellbild")
plt.show()
w=3840
h=2748
'Kennzahlen für räumliche Inhomogenität'
s_ydark_2 = 1/((w*h)-1) * np.sum((y_dark - np.mean(y_dark))**2)
s_y50_2 = 1/((w*h)-1) * np.sum((y_50 - np.mean(y_50))**2)


print(s_ydark_2, s_y50_2)

'pixelweise zeitliche Rauschvarianz'
sigma_s_2_bright = np.zeros(A3_bright[0].shape)
for i in range(15):
    sigma_s_2_bright = sigma_s_2_bright + (A3_bright[i] - y_50)**2
sigma_s_2_bright = sigma_s_2_bright/15

sigma_s_2_dark = np.zeros(A3_dark[0].shape)
for i in range(15):
    sigma_s_2_dark = sigma_s_2_dark + (A3_dark[i] - y_dark)**2
sigma_s_2_dark = sigma_s_2_dark/15

'Durchschnittliche Rauschvarianz über alle Pixel'
sigma_y_2_stack_bright = np.mean(sigma_s_2_bright)
sigma_y_2_stack_dark = np.mean(sigma_s_2_dark)

'Korrigierte räumliche Varianz'
s_50 = s_y50_2 - sigma_y_2_stack_bright/16
s_dark = s_ydark_2 - sigma_y_2_stack_dark/16

u_y_dark_A3 = np.mean(y_dark)
u_y_50_A3 = np.mean(y_50)

DSNU = np.sqrt(s_dark)/K
PRNU = np.sqrt(s_50 - s_dark)/(u_y_50_A3 - u_y_dark_A3)
print('DSNU = ', DSNU, 'e^-')
print('PRNU = ', PRNU*100,'%')

'Spektogramme'
y_dark_norm = y_dark - np.mean(y_dark)
y_50_norm = y_50 - np.mean(y_50)

# plt.imshow(y_50_norm,cmap='gray')


#
#
# gen = np.zeros((3840, 2748))
# for i, x in enumerate(gen):
#     if i%2!=0:
#         gen[i] = x + 255

# y_dark_norm = np.transpose(gen)




'Horizontales Spektroggramm DSNU'
Y_m_v_dark = np.fft.fft(y_dark_norm, axis=0)
p_v_m_dark = np.zeros(Y_m_v_dark.shape[1])
for i in range(Y_m_v_dark.shape[0]):
    p_v_m_dark = p_v_m_dark + (Y_m_v_dark[i,:] * np.transpose(np.conj(Y_m_v_dark[i,:])))


p_v_m_dark = np.sqrt(p_v_m_dark * 1/Y_m_v_dark.shape[0])[:1920]
freq = np.fft.fftfreq(Y_m_v_dark.shape[-1])[:1920]

plt.plot(freq,np.abs(p_v_m_dark),label='Spektrogramm')
plt.yscale('log')

#plt.ylim(ymax=100, ymin=0.01)
plt.autoscale()
# plt.xlim(xmin=0, xmax=1)
plt.xlim(xmin=0, xmax=0.5)
plt.title('Horizontales Spektroggramm DSNU (U-Kugel)')
plt.xlabel('frequency in cycles/pixel')
plt.ylabel('standard deviation in DN')
plt.hlines(y=DSNU, xmin=0, xmax=1, color='black', linestyles = '--', label='DSNU')
plt.hlines(y=sigma_y_2_stack_dark, xmin=0, xmax=1, color='black', linestyles='-.', label='sigma_y_2_stack_dark')
plt.legend()
plt.show()

'Horizontales Spektroggramm PRNU'

plt.imshow(y_50_norm, cmap="gray")
plt.show()

Y_m_v_bright = np.fft.fft(y_50_norm, axis=1)
p_v_m_bright = np.zeros(Y_m_v_bright.shape[1])
for i in range(Y_m_v_bright.shape[0]):
    p_v_m_bright = p_v_m_bright + (Y_m_v_bright[i,:] * np.transpose(np.conj(Y_m_v_bright[i,:])))


p_v_m_bright = np.sqrt(p_v_m_bright * 1/Y_m_v_bright.shape[0])[:1920]
freq = np.fft.fftfreq(Y_m_v_bright.shape[-1])[:1920]


p_v_m_bright = p_v_m_bright/u_y_50_A3

plt.plot(freq,np.abs(p_v_m_bright),label='Spektogramm')
plt.yscale('log')
plt.autoscale()
plt.xlim(xmin=0, xmax=0.49)
# plt.ylim(ymax=0t ymin=0.01)
plt.title('Horizontales Spektroggramm PRNU (U-Kugel)')
plt.xlabel('frequency in cycles/pixel')
plt.ylabel('standard deviation in %')
plt.hlines(y=PRNU, xmin=0, xmax=1, color='black', linestyles = '--',label='PRNU')
plt.hlines(y=sigma_y_2_stack_bright, xmin=0, xmax=1, color='black', linestyles='-.', label='sigma_y_2_stack_bright')
plt.legend()
plt.show()

'Vertikales Spektroggramm DSNU'
Y_n_v_dark = np.fft.fft(y_dark_norm, axis=1)
p_v_n_dark = np.zeros(Y_n_v_dark.shape[0])
for i in range(Y_n_v_dark.shape[1]):
    p_v_n_dark = p_v_n_dark + (Y_n_v_dark[:,i] * np.transpose(np.conj(Y_n_v_dark[:,i])))

p_v_n_dark = np.sqrt(p_v_n_dark * 1/Y_n_v_dark.shape[0])[:1374]
freq = np.fft.fftfreq(Y_n_v_dark.shape[0])[:1374]

plt.plot(freq,np.abs(p_v_n_dark), label='Spektogramm')
plt.yscale('log')

#plt.ylim(ymax=10, ymin=0.0001)

plt.autoscale()
plt.xlim(xmin=0, xmax=0.5)
plt.title('Vertikales Spektroggramm DSNU (U-Kugel)')
plt.xlabel('frequency in cycles/pixel')
plt.ylabel('standard deviation in DN')
plt.hlines(y=DSNU, xmin=0, xmax=1, color='black', linestyles = '--', label='DSNU')
plt.hlines(y=sigma_y_2_stack_dark, xmin=0, xmax=1, color='black', linestyles='-.', label='sigma_y_2_stack_dark')
plt.legend()
plt.show()

'Vertikales Spektroggramm PRNU'
Y_n_v_bright = np.fft.fft(y_50_norm, axis=1)
p_v_n_bright = np.zeros(Y_n_v_bright.shape[0])
for i in range(Y_n_v_bright.shape[1]):
    p_v_n_bright = p_v_n_bright + (Y_n_v_bright[:,i] * np.transpose(np.conj(Y_n_v_bright[:,i])))

p_v_n_bright = np.sqrt(p_v_n_bright * 1/Y_n_v_bright.shape[0])[:1374]
freq = np.fft.fftfreq(Y_n_v_bright.shape[0])[:1374]

plt.plot(freq,np.abs(p_v_n_bright), label='Spektogramm')
plt.yscale('log')
plt.autoscale()
plt.xlim(xmin=0, xmax=0.49)
#plt.ylim(ymax=1000, ymin=0.01)
plt.title('Vertikales Spektroggramm PRNU (U-Kugel)')
plt.xlabel('frequency in cycles/pixel')
plt.ylabel('standard deviation in %')
plt.hlines(y=PRNU, xmin=0, xmax=1, color='black', linestyles = '--', label='PRNU')
plt.hlines(y=sigma_y_2_stack_bright, xmin=0, xmax=1, color='black', linestyles='-.', label='sigma_y_2_stack_brigth')
plt.legend()
plt.show()

## Aufgabe 4

plt.imshow(y_dark,cmap="gray")
plt.show()

lowpass_img = gaussian_filter(y_50_norm, sigma=13)
highpass_img = y_50 - lowpass_img
highpass_img = highpass_img + np.abs(np.min(highpass_img))


'Histogramme'
plt.hist(y_dark.flatten()-np.mean(y_dark.flatten()), 255)
plt.yscale('log')

plt.title('Histogramm DSNU')
plt.vlines(9, 0, 10000, linestyles="dotted", colors="red")
plt.xlabel('Deviation from mean in DN')
plt.ylabel('Number of pixel/bin')
plt.legend(["Schwellwert", "Daten"])
plt.show()

plt.hist(highpass_img.flatten()-np.mean(highpass_img.flatten()), 255)
plt.yscale('log')
plt.ylim(ymin=0.5)

'Gauss Fit für Histogram'
y_histogram, x_histogram = np.histogram(highpass_img.flatten()-np.mean(highpass_img.flatten()), 255)
parameters, covariance = curve_fit(Gauss, x_histogram[:-1], y_histogram)
fit_A = parameters[0]
fit_B = parameters[1]
fit_y = Gauss(x_histogram, fit_A, fit_B)
# plt.plot(x_histogram,fit_y, '--', label='fit')

plt.title('Histogramm PRNU')
plt.vlines(-13.7, 0, 10000, linestyles="dotted", colors="red")
plt.xlabel('Deviation from mean in %')
plt.ylabel('Number of pixel/bin')
plt.legend(["Schwellwert", "Daten"])
plt.show()



defect_pixel = highpass_img-np.mean(highpass_img)
# circle1 = plt.Circle((np.where(defect_pixel>15)[1],np.where(defect_pixel>15)[0]), 10, color='r', fill=False)
# circle2 = plt.Circle((np.where(defect_pixel<-14.5)[1][0],np.where(defect_pixel<-14.5)[0][0]), 10, color='r', fill=False)
circle3 = plt.Circle((np.where(defect_pixel<-13.7)[1][0],np.where(defect_pixel<-13.8)[0][0]), 10, color='r', fill=False)
circle2 = plt.Circle((np.where(defect_pixel<-13.7)[1][1],np.where(defect_pixel<-13.8)[0][1]), 10, color='r', fill=False)

# circle4 = plt.Circle((np.where(defect_pixel>-13.25)[1][1],np.where(defect_pixel>13.25)[0][1]), 10, color='r', fill=False)

print(circle3, circle2)


lowpass_img_dark = gaussian_filter(y_dark_norm, sigma=13)
highpass_img_dark = y_dark - lowpass_img_dark
highpass_img_dark = highpass_img_dark + np.abs(np.min(highpass_img_dark))


hot_stuck_pixel = highpass_img_dark - np.mean(y_dark)
circledark = plt.Circle((np.where(hot_stuck_pixel>9)[1][0],np.where(hot_stuck_pixel>9)[0][0]), 10, color='r', fill=False)
circledark1 = plt.Circle((np.where(hot_stuck_pixel>9)[1][1],np.where(hot_stuck_pixel>9)[0][1]), 10, color='r', fill=False)
circledark2 = plt.Circle((np.where(hot_stuck_pixel>9)[1][2],np.where(hot_stuck_pixel>9)[0][2]), 10, color='r', fill=False)


fig, ax = plt.subplots()
fig = plt.imshow(y_dark_norm, cmap='gray')
ax.add_patch(circledark)
ax.add_patch(circledark1)
ax.add_patch(circledark2)
plt.show()



fig, ax = plt.subplots()
fig = plt.imshow(y_50_norm, cmap='gray')
# ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
# ax.add_patch(circle4)
plt.show()






# Aufgabe 5

def getdarkorwhiteimg(path):
    img_names = [f for f in listdir(path) if isfile(join(path, f))]
    images = []
    for name in img_names:
        images.append(importimage(path + name))

    img_mean = np.zeros(images[0].shape)
    for i in range(len(images)):
        img_mean = img_mean + images[i]
    img_mean = img_mean / len(images)
    return img_mean


def interpolat_dark_image(y_dark_t1, y_dark_t2, t1, t2, texp):
    y_dark_texp = (t2 - texp) / (t2 - t1) * y_dark_t1 + (texp - t1) / (t2 - t1) * y_dark_t2
    return y_dark_texp




'Dunkel- und Hellbilder importieren'
# y_dark_t4500 = getdarkorwhiteimg('Test_Aufgabe_5_neu/Dunkelbild/B_50/')
# y_dark_t1000 = getdarkorwhiteimg('Test_Aufgabe_5_neu/Dunkelbild/B_5000/')
# y_50 = getdarkorwhiteimg('Test_Aufgabe_5_neu/Optik/Neu/')


ydark_2 = importimage(r"C:\Users\ruwen\Desktop\optik_mme\u-kugel\neu\aufgabe5\dunkel_belichtungszeit_0408ms.bmp")

'Dunkelbild interpolieren'
y_dark_texp = interpolat_dark_image(y_dark, ydark_2, 11000, 408, 40000)

plt.imshow(y_dark_texp, cmap="gray")
plt.show()

# y_dark_texp = interpolat_dark_image(y_dark_t4500, y_dark_t1000, 50, 5000, 200000)

'Weißbild korrigieren'
y_50i = y_50 - y_dark_texp

'Zu korrigierendes Bild importieren'
y = importimage(r'C:\Users\ruwen\Desktop\optik_mme\u-kugel\neu\aufgabe5\testbild_40ms_f1-4.bmp')

yi = np.mean(y_50i) * (y - y_dark_texp) / y_50i



# fig, ax = plt.subplots(2)
# ax[0].imshow(y, cmap='gray')
# ax[1].imshow(yi, cmap='gray')
# plt.show()




plt.figure(30)
# Nicht Korrigiert
plt.imshow(y, cmap='gray')
plt.imsave("unkali.png", y, cmap="gray")
plt.show()
plt.figure(31)
# Korrigiert
plt.imshow(yi, cmap='gray')
plt.imsave("kali.png", yi, cmap="gray")
plt.show()


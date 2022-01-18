# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:03:32 2022

@author: Milan Kaiser
"""

import matplotlib.pyplot as plt	# for plots
from PIL import Image
import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import imageio
import os
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

def mittelgrau(img1, img2):
    'Formel Vorl. 3 Folie 13/23'
    x = 1/(2*img1.shape[0]*img1.shape[1]) * np.sum(img1)+1/(2*img1.shape[0]*img1.shape[1] )*np.sum(img2)
    return x

def zeitrauschen(img1, img2):
    'Formel Vorl. 3 Folie 13/23'
    x = 1/(2*img1.shape[0]*img1.shape[1]) * np.sum((img1-img2)**2)
    return x

## Aufgabe 1
'Parameter eingeben'
A = 23.04e-12               # Pixelfläche [m^2]
wellenlaenge = 475*10**-9   # Wellenlänge [m]
E = 7.82e-6/3.352e-9/1699/0.28                      # Bestrahlungsstärke [W/m^2]  Formel: Messwert/Kalibrierwert/k_skoptisch/V(lambda)
print(E)
t = np.array([1,2,3,4,5,6,7,8,9,10]) # Belichtungszeit [ms]
t_exp = t*1e-3

u_p = 5.034e24 * A * E * t_exp * wellenlaenge


basepath=r"C:\Users\Chris\Desktop\bildgebende_optische_systeme\optik_mme\u-kugel\neu\hell"
'Bilder importieren'
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

basepath=r"C:\Users\Chris\Desktop\bildgebende_optische_systeme\optik_mme\u-kugel\neu\dunkel"

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

'Varianz Berechnen'
# var = np.array([np.var(A_40_B_1000_2),np.var(A_40_B_2000_2),np.var(A_40_B_3000_2),np.var(A_40_B_4000_2),np.var(A_40_B_5000_2),np.var(A_40_B_6000_2),np.var(A_40_B_7000_2),np.var(A_40_B_8000_2),np.var(A_40_B_9000_2),np.var(A_40_B_10000_2)])
# plt.plot(t_exp,var, "o--")
# plt.show()
# var = np.array([np.var(A_40_B_1000_ABG_2),np.var(A_40_B_2000_ABG_2),np.var(A_40_B_3000_ABG_2),np.var(A_40_B_4000_ABG_2),np.var(A_40_B_5000_ABG_2),np.var(A_40_B_6000_ABG_2),np.var(A_40_B_7000_ABG_2),np.var(A_40_B_8000_ABG_2),np.var(A_40_B_9000_ABG_2),np.var(A_40_B_10000_ABG_2)])
# plt.plot(t_exp,var, "o--")
# plt.show()
# var = np.array([np.var(A_40_B_1000_ABG_1),np.var(A_40_B_2000_ABG_1),np.var(A_40_B_3000_ABG_1),np.var(A_40_B_4000_ABG_1),np.var(A_40_B_5000_ABG_1),np.var(A_40_B_6000_ABG_1),np.var(A_40_B_7000_ABG_1),np.var(A_40_B_8000_ABG_1),np.var(A_40_B_9000_ABG_1),np.var(A_40_B_10000_ABG_1)])
# plt.plot(t_exp,var, "o--")
# plt.show()

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
x_sens = np.linspace(1,80000,10)
res_sens = stats.linregress(u_p, delta_u)
plt.plot(x_sens,res_sens.intercept + res_sens.slope*x_sens, label='fit')
plt.title('Sensitivität')
plt.xlabel('irradiation in photons/pixel')
plt.ylabel('gray value - dark value µ_y−µ_ydark')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.legend()
plt.grid()
plt.show()

'Photonentransferkurve Plotten'
plt.plot(delta_u,delta_s,'o', label='Data')
x_photo = np.linspace(1,230,10)
res_sens = stats.linregress(delta_u[:7],delta_s[:7])
plt.plot(x_photo,res_sens.intercept + res_sens.slope*x_photo, label='fit')
plt.title('Photonentransfer')
plt.xlabel('gray value - dark value µ_y−µ_ydark')
plt.ylabel('variance gray value σ_y^2−σ_ydark^2')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.legend()
plt.grid()
plt.show()

'Kennwerte Bestimmen'
'Bei K und R nur 70% der Werte nutzen'
R = stats.linregress(u_p[:7], delta_u[:7])[0]
K = stats.linregress(delta_u[:7], delta_s[:7])[0]
eta = R/K
s_q = 1/12
s_d = (s_y_dark[0]-s_q)/K**2
'Kennwerte ausgeben'
print('R = ', + R)
print('K = ', + K)
print('eta = ', eta*100,'%')
print('σ_d = ', + s_d)

## Aufgabe 2
'SNR-Kurve'
SNR = (eta * u_p)/np.sqrt(s_d+s_q/K**2 + eta * u_p)
'Data plotten:'
plt.loglog(u_p,SNR,'o', label='Data')
x_snr = np.linspace(1,np.max(u_p),10)
'theor. limit plotten:'
plt.loglog(x_snr,np.sqrt(x_snr), label='theor. limit')
plt.title('SNR-Kurve')
plt.xlabel('irradiation in photons/pixel')
plt.ylabel('SNR')
plt.grid(True, which="both", ls="-")
plt.xlim(xmin=1)
plt.ylim(ymin=0.1)

'u_p min und u_p sat, senkrechte Linien plotten'
plt.vlines(x=u_p[8], ymin=0, ymax=10000, color='black')
plt.text(u_p[8],0.5,'Saturation',rotation=90)
u_pmin = 1/eta*(np.sqrt(s_y_dark[0])/K+0.5)
plt.vlines(x=u_pmin, ymin=0, ymax=10000, color='black')
plt.text(u_pmin,0.5,'threshold',rotation=90)
DR = 20 * np.log10(u_p[8]/u_pmin)

'Fit plotten, kubisches Polynom'
snr_fit = np.polyfit(u_p, SNR, 2)
x_snr = np.linspace(0,900000,100)
y_snr = snr_fit[0]*x_snr**2 + snr_fit[1]*x_snr**1 + snr_fit[2]*x_snr**0 
plt.loglog(x_snr, y_snr)
plt.legend()
plt.show()

## Aufgabe 3
'Importieren der Bilder'
path_img_bright = 'Test_Aufgabe_3/Hell/'
path_img_dark = 'Test_Aufgabe_3/Dunkel/'
imgnames_bright = [f for f in listdir(path_img_bright) if isfile(join(path_img_bright, f))]
imgnames_dark = [f for f in listdir(path_img_dark) if isfile(join(path_img_dark, f))]

A3_bright = []
A3_dark = []

for name in imgnames_bright:
     A3_bright.append(importimage(path_img_bright + name))
for name in imgnames_dark:
     A3_dark.append(importimage(path_img_dark + name))

'Dunkelbild'
y_dark = np.zeros(A3_dark[0].shape)
for i in range(16) : 
    y_dark = y_dark + A3_dark[i]
y_dark = y_dark/16

'Hellbild'
y_50 = np.zeros(A3_bright[0].shape)
for i in range(16) : 
    y_50 = y_50 + A3_bright[i]
y_50 = y_50/16

'Kennzahlen für räumliche Inhomogenität'
s_ydark_2 = 1/(480*640-1) * np.sum((y_dark - np.mean(y_dark))**2)
s_y50_2 = 1/(480*640-1) * np.sum((y_50 - np.mean(y_50))**2)

'pixelweise zeitliche Rauschvarianz'
sigma_s_2_bright = np.zeros(A3_bright[0].shape)
for i in range(16):
    sigma_s_2_bright = sigma_s_2_bright + (A3_bright[i] - y_50)**2
sigma_s_2_bright = sigma_s_2_bright/15

sigma_s_2_dark = np.zeros(A3_dark[0].shape)
for i in range(16):
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

'Horizontales Spektroggramm DSNU'
Y_m_v_dark = np.fft.fft(y_dark_norm, axis=0)
p_v_m_dark = np.zeros(Y_m_v_dark.shape[1])
for i in range(Y_m_v_dark.shape[0]):
    p_v_m_dark = p_v_m_dark + (Y_m_v_dark[i,:] * np.transpose(np.conj(Y_m_v_dark[i,:])))

p_v_m_dark = np.sqrt(p_v_m_dark * 1/Y_m_v_dark.shape[0])[:320]
freq = np.fft.fftfreq(Y_m_v_dark.shape[-1])[:320]

plt.plot(freq,np.abs(p_v_m_dark))
plt.yscale('log')
plt.xlim(xmin=0, xmax=0.5)
plt.ylim(ymax=1000, ymin=0.01)
plt.title('Horizontales Spektroggramm DSNU')
plt.show()

'Horizontales Spektroggramm PRNU'
Y_m_v_bright = np.fft.fft(y_50_norm, axis=0)
p_v_m_bright = np.zeros(Y_m_v_bright.shape[1])
for i in range(Y_m_v_bright.shape[0]):
    p_v_m_bright = p_v_m_bright + (Y_m_v_bright[i,:] * np.transpose(np.conj(Y_m_v_bright[i,:])))

p_v_m_bright = np.sqrt(p_v_m_bright * 1/Y_m_v_bright.shape[0])[:320]
freq = np.fft.fftfreq(Y_m_v_bright.shape[-1])[:320]

plt.plot(freq,np.abs(p_v_m_bright))
plt.yscale('log')
plt.xlim(xmin=0, xmax=0.5)
plt.ylim(ymax=1000, ymin=0.01)
plt.title('Horizontales Spektroggramm PRNU')
plt.show()

'Vertikales Spektroggramm DSNU'
Y_n_v_dark = np.fft.fft(y_dark_norm, axis=1)
p_v_n_dark = np.zeros(Y_n_v_dark.shape[0])
for i in range(Y_n_v_dark.shape[1]):
    p_v_n_dark = p_v_n_dark + (Y_n_v_dark[:,i] * np.transpose(np.conj(Y_n_v_dark[:,i])))

p_v_n_dark = np.sqrt(p_v_n_dark * 1/Y_n_v_dark.shape[0])[:240]
freq = np.fft.fftfreq(Y_n_v_dark.shape[0])[:240]

plt.plot(freq,np.abs(p_v_n_dark))
plt.yscale('log')
plt.xlim(xmin=0, xmax=0.5)
plt.ylim(ymax=1000, ymin=0.01)
plt.title('Vertikales Spektroggramm DSNU')
plt.show()

'Vertikales Spektroggramm PRNU'
Y_n_v_bright = np.fft.fft(y_50_norm, axis=1)
p_v_n_bright = np.zeros(Y_n_v_bright.shape[0])
for i in range(Y_n_v_bright.shape[1]):
    p_v_n_bright = p_v_n_bright + (Y_n_v_bright[:,i] * np.transpose(np.conj(Y_n_v_bright[:,i])))

p_v_n_bright = np.sqrt(p_v_n_bright * 1/Y_n_v_bright.shape[0])[:240]
freq = np.fft.fftfreq(Y_n_v_bright.shape[0])[:240]

plt.plot(freq,np.abs(p_v_n_bright))
plt.yscale('log')
plt.xlim(xmin=0, xmax=0.5)
plt.ylim(ymax=1000, ymin=0.01)
plt.title('Vertikales Spektroggramm PRNU')
plt.show()

## Aufgabe 4
lowpass_img = gaussian_filter(y_50, sigma=7)
highpass_img = y_50 - lowpass_img
highpass_img = highpass_img + np.abs(np.min(highpass_img))
# plt.imshow(highpass_img, cmap='gray')
# plt.show()

'Histogramme'
plt.hist(y_dark_norm.flatten()-np.mean(y_dark_norm.flatten()), 255)
plt.yscale('log')

plt.title('Histogramm DSNU')
plt.show()

plt.hist(highpass_img.flatten()-np.mean(highpass_img.flatten()), 255)
plt.yscale('log')
# plt.xlim(xmin=-1, xmax=1)
plt.title('Histogramm PRNU')
plt.show()


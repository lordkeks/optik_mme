from PIL import Image
import numpy as np


image = Image.open(r"C:\Users\ruwen\OneDrive\Desktop\optik_mme\coding\opencv.png")

imgMatrix = np.asarray(image)
std = imgMatrix.std()



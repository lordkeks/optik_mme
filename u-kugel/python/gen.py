"""
Name : gen.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 19.01.2022 11:44
Desc:
"""

import numpy as np
import matplotlib.pyplot as plt


gen = np.zeros((3840, 2748))
for i, x in enumerate(gen):
    if i%2!=0:
        gen[i] = x + 255
plt.imshow(gen)
plt.show()

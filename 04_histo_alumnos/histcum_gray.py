

import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the input image")
args = vars(ap.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)

#histogram is computed
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
histcum = np.cumsum(hist)

#se genera una figura para mostrar los resultados con matplotlib
fig=plt.figure(figsize=(14,5))
#se maqueta el diseño del grafico
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
#se dibuja la imagen original
ax1.imshow(image, cmap='gray')
ax1.set_title('Original image')
#se dibuja el histograma
ax2.plot(histcum)
ax2.set_title('Cumulative Histogram')

plt.show()
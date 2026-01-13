import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

# argument parser`
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# original image
image = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
b,g,r = cv2.split(image)

#histogram is computed
histb = cv2.calcHist([b], [0], None, [256], [0, 256])
histg = cv2.calcHist([g], [0], None, [256], [0, 256])
histr = cv2.calcHist([r], [0], None, [256], [0, 256])

fig = plt.figure(figsize=(14,14))
#se maqueta el diseño del grafico
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,3,4)
ax3=fig.add_subplot(2,3,5)
ax4=fig.add_subplot(2,3,6)

#se dibuja la imagen original
ax1.imshow(image_RGB)
ax1.set_title('Original image')

#se dibuja el histograma
ax2.plot(histr, color='r')
ax2.set_title('Histogram of Red Channel')

ax3.plot(histg, color='g')
ax3.set_title('Histogram of Green Channel')

ax4.plot(histb, color='b')
ax4.set_title('Histogram of Blue Channel')

plt.show()
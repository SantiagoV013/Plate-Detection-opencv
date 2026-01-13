import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# Main program ***************************************
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the input image")
ap.add_argument("-o", "--output", required = False, help = "Name to the output image")
arg = ap.parse_args()
args = vars(arg)

#reading the input image as it is
input_image = cv2.imread(args["image"],cv2.IMREAD_UNCHANGED)
#number of channels
if(len(input_image.shape) == 2):
    n_channels = 1
else:
    n_channels = 3

if n_channels == 1:
    image = input_image
else:
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)    

# equalization is applyed to the input image
if n_channels == 1:
    image_eq = cv2.equalizeHist(image)
else:
    #equalization is performed based on a HSV color space (using V channel)
    H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    eq_V = cv2.equalizeHist(V)
    image_eq = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)    

#se genera una figura para mostrar los resultados con matplotlib
fig=plt.figure(figsize=(14,10))
#se maqueta el diseño del grafico
ax1=fig.add_subplot(2,3,1)
ax2=fig.add_subplot(2,3,2)
ax3=fig.add_subplot(2,3,3)
ax4=fig.add_subplot(2,3,4)
ax5=fig.add_subplot(2,3,5)
ax6=fig.add_subplot(2,3,6)


#se dibuja el histograma de imagen original
if n_channels == 1:
    #se dibuja la imagen original
    img = cv2.merge([image,image,image])
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original image')
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ax2.plot(hist)
    histcum = np.cumsum(hist)
    ax3.plot(histcum)
else:
    #se dibuja la imagen original
    ax1.imshow(image)
    ax1.set_title('Original image')
    colors = ("r", "g", "b")
    chans = cv2.split(image)
    for (chan, color) in zip (chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        histcum = np.cumsum(hist)
        ax2.plot(hist, color = color)
        ax3.plot(histcum, color = color)
        
ax2.set_title('Histogram')
ax3.set_title('Cumulative Histogram')

#se dibuja el histograma de la imagen corregida
if n_channels == 1:
    #se dibuja la imagen corregida 
    img_eq = cv2.merge([image_eq,image_eq,image_eq])
    ax4.imshow(img_eq, cmap='gray')
    ax4.set_title('Equalization')
    hist = cv2.calcHist([image_eq], [0], None, [256], [0, 256])
    ax5.plot(hist)
    histcum = np.cumsum(hist)
    ax6.plot(histcum)
else:
    #se dibuja la imagen corregida
    ax4.imshow(image_eq)
    ax4.set_title('Equalization')
    colors = ("r", "g", "b")
    chans = cv2.split(image_eq)
    for (chan, color) in zip (chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        ax5.plot(hist, color = color)
        histcum = np.cumsum(hist)
        ax6.plot(histcum, color = color)

ax5.set_title('Equalization Histogram')
ax6.set_title('Cumulative Histogram')
plt.show()




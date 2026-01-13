# Ivan Olmos
# Curso vision

import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

# functions ******************************************
def min(size, histacum, qlow):
    v = size * qlow
    #return indices where the condicion is satisfied
    pos = np.where(histacum >= v)[0]
    #return the index with the minimum value in the array
    #return np.min(pos)
    return (pos[0])

def max(size, histacum, qhigh):
    v = size * (1 - qhigh)
    #return indices where the condicion is satisfied
    pos = np.where(histacum <= v)[0]
    #return the index with the maximum value in the array
    #return np.max(pos)
    return (pos[pos.size - 1])

# Main program ***************************************
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the input image")
ap.add_argument("-b", "--low", required = True, help = "q_low value")
ap.add_argument("-a", "--high", required = True, help = "q_high value")
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
    imageGray = image
else:
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)    
    # it is calculated the gray-scale image from the input image
    imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#argument values are readed
# 0 <= q_low, q_high <= 1, q_low + q_high <= 1
q_low = float(args["low"])
q_high = float(args["high"])

#original histogram is computed
hist = cv2.calcHist([imageGray], [0], None, [256], [0, 256])
#cumulative histogram
histcum = np.cumsum(hist)
#size of the image
size = imageGray.shape[0] * imageGray.shape[1]

#alow and ahigh values are computed
alow = min(size,histcum,q_low)
ahigh = max(size,histcum,q_high)
amin = 0
amax = 255

# Defining the function
c = (amax - amin) / (ahigh - alow)
# Autocontrast without optimization
autoC = lambda x: amin if x <= alow else (amax if x >= ahigh else amin + (x - alow)*c)
# apply lambda function to each pixel in the input image
#image_autoc = np.array(np.vectorize(autoC)(image),dtype='uint8')

# Autocontrast with optimization
x = np.arange(256)
f = np.array(np.vectorize(autoC)(x),dtype='uint8')
map = lambda p: f[p]
image_autoc = np.array(np.vectorize(map)(image))

#se guarda el archivo de salida
if arg.output:
    if n_channels == 3:
        cv2.imwrite(args["output"], cv2.cvtColor(image_autoc,cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(args["output"], image_autoc)
        
#se genera una figura para mostrar los resultados con matplotlib
fig=plt.figure(figsize=(14,10))
#se maqueta el diseño del grafico
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)

#se dibuja la imagen y el histograma de imagen original
if n_channels == 1:
    #se dibuja la imagen original
    img = cv2.merge([image,image,image])   
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original image')
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ax2.plot(hist)
else:
    ax1.imshow(image)
    ax1.set_title('Original image')
    colors = ("r", "g", "b")
    chans = cv2.split(image)
    for (chan, color) in zip (chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        ax2.plot(hist, color = color)
ax2.set_title('Histogram')

#se dibuja el histograma de la imagen corregida
if n_channels == 1:
    #se dibuja la imagen corregida
    img_c = cv2.merge([image_autoc, image_autoc, image_autoc])
    ax3.imshow(img_c, cmap='gray')
    ax3.set_title('Autocontrast')
    hist = cv2.calcHist([image_autoc], [0], None, [256], [0, 256])
    ax4.plot(hist)
else:
    #se dibuja la imagen corregida
    ax3.imshow(image_autoc)
    ax3.set_title('Autocontrast')
    colors = ("r", "g", "b")
    chans = cv2.split(image_autoc)
    for (chan, color) in zip (chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        ax4.plot(hist, color = color)

ax4.set_title('Autocontrast Histogram')

plt.show()

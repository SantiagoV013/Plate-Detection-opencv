import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--image",
    default="/Users/svo/Documents/qoop.Ai_GitHub/OpenCV/04_histo_alumnos/input_low.jpg",
    help="Path to the input image"
)
ap.add_argument("-o", "--output", required=False, help="Name to the output image")
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


#autocontrast is computed
a_low = image.min()
a_high = image.max()
c = 255 / (a_high - a_low)
image_autoc = np.array((image - a_low) * c, dtype='uint8')
hist_autoc = cv2.calcHist([image_autoc], [0], None, [256], [0, 256])

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


#se dibuja el histograma de imagen original
if n_channels == 1:
    #se dibuja la imagen original
    img = cv2.merge([image, image, image])
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original image')
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ax2.plot(hist)
else:
    #se dibuja la imagen original
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
    ax3.imshow(image_autoc, cmap='gray')
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

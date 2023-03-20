import cv2
import numpy as np
import matplotlib.pyplot as plt

# imagen original
img = cv2.imread('ronal.jpg')

img_reducida = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

img_gris = cv2.cvtColor(img_reducida, cv2.COLOR_BGR2GRAY)

img_invertida = cv2.flip(img_reducida, 1)

img_bordes = cv2.Canny(img_reducida, 100, 200)

fig, axs = plt.subplots(4, 2, figsize=(10, 30))


axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Imagen a color')
axs[0, 1].imshow(cv2.cvtColor(img_reducida, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Imagen reducida')

axs[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Imagen original')
axs[1, 1].imshow(img_gris, cmap='gray')
axs[1, 1].set_title('Imagen en escala de grises')

axs[2, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[2, 0].set_title('Imagen original')
axs[2, 1].imshow(cv2.cvtColor(img_invertida, cv2.COLOR_BGR2RGB))
axs[2, 1].set_title('Imagen invertida')

axs[3, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[3, 0].set_title('Imagen original')
axs[3, 1].imshow(img_bordes, cmap='gray')
axs[3, 1].set_title('Imagen con filtro de detección de bordes')

# Ajustar el espacio entre subplots verticalmente
plt.subplots_adjust(hspace=0.5)

plt.show()
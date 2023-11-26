import pixels2svg
from PIL import Image
from math import floor, ceil
import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.signal import convolve2d


def convolution(a, b):
    s = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            s += a[i][j] * b[i][j]
    return s

filter = [
    [-1, -1, 0, 0, 0],
    [0, -1, -1, -1, 0],
    [0, -1, 9, -1, 0],
    [0, -1, -1, -1, 0],
    [0, 0, 0, 0, 0]
]

div = sum(sum(filter, []))
if div == 0:
    div = 1

img = Image.open('./foto.jpg')
pixels = img.load()

# Создаем два графика (subplot) для отображения изображений рядом
plt.subplot(121)
plt.imshow(img)

img_convolved = img.copy()
pixels2 = img_convolved.load()

start_time=time.time()


for i in range(floor(len(filter)/2), img.height - floor(len(filter)/2)):
    for j in range(floor(len(filter[0])/2), img.width - floor(len(filter[0])/2)):
        matrR = []
        matrG = []
        matrB = []
        for n in range(-floor(len(filter)/2), ceil(len(filter)/2)):
            rowR = []
            rowG = []
            rowB = []
            for m in range(-floor(len(filter[0])/2), ceil(len(filter[0])/2)):
                px, py = j + m, i + n
                r, g, b = pixels[px, py] if 0 <= px < img.width and 0 <= py < img.height else (0, 0, 0)
                rowR.append(r)
                rowG.append(g)
                rowB.append(b)
            matrR.append(rowR)
            matrG.append(rowG)
            matrB.append(rowB)

        r = np.clip(round(convolution(matrR, filter) / div), 0, 255)
        g = np.clip(round(convolution(matrG, filter) / div), 0, 255)
        b = np.clip(round(convolution(matrB, filter) / div), 0, 255)

        pixels2[j, i] = (r, g, b)

end_time=time.time()-start_time
st_time1=time.time()
for i in range(floor(len(filter)/2), img.height - floor(len(filter)/2)):
    for j in range(floor(len(filter[0])/2), img.width - floor(len(filter[0])/2)):
        matrR = np.array([[pixels[j+m, i+n][0] if 0 <= j+m < img.width and 0 <= i+n < img.height else 0 for m in range(-floor(len(filter[0])/2), ceil(len(filter[0])/2))]])
        matrG = np.array([[pixels[j+m, i+n][1] if 0 <= j+m < img.width and 0 <= i+n < img.height else 0 for m in range(-floor(len(filter[0])/2), ceil(len(filter[0])/2))]])
        matrB = np.array([[pixels[j+m, i+n][2] if 0 <= j+m < img.width and 0 <= i+n < img.height else 0 for m in range(-floor(len(filter[0])/2), ceil(len(filter[0])/2))]])

        r = np.clip(round(convolve2d(matrR, filter, mode='valid')[0, 0] / div), 0, 255)
        g = np.clip(round(convolve2d(matrG, filter, mode='valid')[0, 0] / div), 0, 255)
        b = np.clip(round(convolve2d(matrB, filter, mode='valid')[0, 0] / div), 0, 255)

        pixels2[j, i] = (r, g, b)

end_time1=time.time()-st_time1

# Выводим второе изображение (отфильтрованное) на втором графике
plt.subplot(122)
plt.imshow(img_convolved)
plt.show()

print("Время работы вложенных циклов: ", end_time)
print("Время работы библиотечных функций: ", end_time1)

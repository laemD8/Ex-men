import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from kneed import KneeLocator
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

#                Pontificia Universidad Javeriana
#               Procesamiento de imágenes y visión
#                            Exámen Final
#               Laura Alejandra Estupiñan Martínez

class bandera:
    def __init__(self, path_file):  # Ruta imagen
        self.image = cv2.imread(path_file)

    def recreate_image(centers, labels, rows, cols):
        d = centers.shape[1]
        image_clusters = np.zeros((rows, cols, d))
        label_idx = 0
        for i in range(rows):
            for j in range(cols):
                image_clusters[i][j] = centers[labels[label_idx]]
                label_idx += 1
        return image_clusters

    def colores(self):
        image_RGB = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image_RGB = np.array(image_RGB, dtype=np.float64) / 255
        rows, cols, ch = image_RGB.shape
        image_RGB = image_RGB.reshape((rows * cols, ch))
        l = []
        for i in range(1, 4):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(image_RGB)
            n = kmeans.inertia_
            l.append(n)
        print(l)
        x = range(1, len(l)+1)
        k = KneeLocator(x, l)
        plt.plot(list(np.arange(1, 4)), l)
        plt.show()
        return k.knee, image_RGB


    def porcentaje(self, n_colors, image_RGB):
        kmeans = KMeans(n_colors)
        j = kmeans.fit(image_RGB)
        labels = kmeans.labels_
        labels = list(labels)
        centroid = kmeans.cluster_centers_
        percent = []
        for i in range(len(centroid)):
            p = labels.count(i)
            p = p / (len(labels))
            percent.append(p)
        print(percent)
        plt.pie(percent, colors=np.array(centroid / 255), labels=np.arange(len(centroid)))
        plt.show()

if __name__ == '__main__':
    flag = input('Ingrese un número de 1 al 5: ')
    path = '/Users/lauestupinan/Desktop'
    image_name = 'flag'+flag+'.png'
    path_file = os.path.join(path, image_name)
    image= bandera(path_file)
    n_colors, image_RGB= image.colores()
    print(n_colors)
    image.porcentaje(n_colors, image_RGB)



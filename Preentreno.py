# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import logging, os

# Configurar los mensajes de advertencia
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Cargar datos de MNIST
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# Explorar los datos
print(X_train.shape, X_train.dtype)
print(Y_train.shape, Y_train.dtype)
print(X_test.shape, X_test.dtype)
print(Y_test.shape, Y_test.dtype)

# Función para mostrar imágenes
def show_image(imagen, title):
    plt.figure()
    plt.suptitle(title)
    plt.imshow(imagen, cmap="Greys")
    plt.show()

# Mostrar las primeras imágenes
for i in range(3):
    title = f"Mostrando imagen X_train[{i}] -- Y_train[{i}] = {Y_train[i]}"
    show_image(X_train[i], title)

# Función para graficar valores de píxeles
def plot_X(X, title, fila, columna):
    plt.title(title)
    plt.plot(X[:, fila, columna])
    plt.show()

# Graficar los valores de un píxel específico
fila, columna = 10, 10  # Cambiar por la fila y columna deseada
features_fila_col = X_train[:, fila, columna]
print(len(np.unique(features_fila_col)))
title = f"Valores en ({fila}, {columna})"
plot_X(X_train, title, fila, columna)

import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import time

class DecisionStump:
    def __init__(self, n_caracteristicas):

        self.caracteristica_index = np.random.randint(0, n_caracteristicas) # Seleccionar una característica al azar. n_caracteristicas es el número total de características disponibles.
        self.umbral = None # Inicializar el umbral a None. Este será determinado durante el entrenamiento.
        self.polaridad = np.random.choice([1, -1]) # Polaridad puede ser 1 o -1, se elige al azar.
        self.alpha = None # Valor de alpha (peso del clasificador), inicializado a None. Este será calculado durante el entrenamiento.
    
    def predict(self, X):
        predicciones = np.ones(X.shape[0]) # Inicializar todas las predicciones a 1.
        if self.polaridad == 1: # Dependiendo de la polaridad, ajustar las predicciones basadas en el umbral y la característica seleccionada.
            predicciones[X[:, self.caracteristica_index] < self.umbral] = -1  # Si la característica es menor que el umbral, la predicción es -1.
        else:
            predicciones[X[:, self.caracteristica_index] > self.umbral] = -1 # Si la característica es mayor que el umbral, la predicción es -1.
        return predicciones

class Adaboost:
    ## Constructor de clase, con número de clasificadores e intentos por clasificador
    def __init__(self, T=5, A=20):
        # Dar valores a los parámetros del clasificador e iniciar la lista de clasificadores débiles vacía
        self.T = T  # Número de clasificadores
        self.A = A  # Intentos por clasificador para encontrar el mejor umbral
        self.lista_clasificadores = []  # Lista para mantener todos los clasificadores débiles    
        
    ## Método para entrenar un clasificador fuerte a partir de clasificadores débiles mediante Adaboost
    def fit(self, X, Y, verbose=False):
        # (número_de_imágenes, 28, 28)
        n_samples, n_caracteristicas = X.shape # Obtener el número de observaciones y de características por observación de X
                                                # n_samples número total de imágenes en el conjunto de datos
                                                # n_caracterisitcas n pixeles(cada píxel es una característica) de cada imagen
        w = np.full(n_samples, (1 / n_samples)) # Iniciar pesos de las observaciones a 1/n_samples
        #w : array de pesos. Cada elemento de w representa el peso inicial de cada imagen en tu conjunto de datos.

        # Bucle de entrenamiento Adaboost # Bucle de entrenamiento Adaboost: desde 1 hasta T repetir
        for t in range(self.T): ## Hasta T clasificadores
            # Iniciar el clasificador con el mayor error posible para asegurar que se actualice en el primer intento
            clf = DecisionStump(n_caracteristicas)
            min_error = float('inf')

            # Buscar el mejor clasificador débil: desde 1 hasta A repetir
            # Calcular predicciones de ese clasificador para todas las observaciones
            for a in range(self.A):
                # Generar un umbral y característica aleatorios para el clasificador
                caracteristica_index = np.random.randint(0, n_caracteristicas) # Crear un nuevo clasificador débil aleatorio
                umbral = np.random.uniform(np.min(X[:, caracteristica_index]), np.max(X[:, caracteristica_index])) 
                polaridad = np.random.choice([1, -1])
                # Actualizar el clasificador débil con estos parámetros aleatorios
                clf.caracteristica_index = caracteristica_index
                clf.umbral = umbral
                clf.polaridad = polaridad
                # Realizar predicciones con el clasificador débil actual
                predicciones = clf.predict(X)

                # Calcular el error ponderado de las predicciones
                # comparar predicciones con los valores deseados 
                # y acumular los pesos de las observaciones mal clasificadas
                error = np.sum(w[Y != predicciones])

                # Actualizar mejor clasificador hasta el momento: el que tenga menor error
                # Si el error es menor que el error mínimo, actualizar el clasificador y el error mínimo
                if error < min_error:
                    min_error = error
                    best_clf = clf

            # Calcular el valor de alfa y las predicciones del mejor clasificador débil
            best_clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))

            # Guardar el mejor clasificador en la lista de clasificadores de Adaboost
            self.lista_clasificadores.append(best_clf)

            # Actualizar pesos de las observaciones en función de las predicciones, los valores deseados y alfa
            # Y Normalizar a 1 los pesos
            predicciones = best_clf.predict(X)
            w *= np.exp(-best_clf.alpha * Y * predicciones)
            w /= np.sum(w)

            if verbose:
                print(f'Ronda {t+1}, Caracteristica {best_clf.caracteristica_index}, Umbral {best_clf.umbral}, Polaridad {best_clf.polaridad}, Error {min_error}, Alpha {best_clf.alpha}')
    
    # Método para obtener una predicción con el clasificador fuerte Adaboost 
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.lista_clasificadores] # Calcular las predicciones de cada clasificador débil para cada input multiplicadas por su alfa
        y_pred = np.sum(clf_preds, axis=0) # Sumar para cada input todas las predicciones ponderadas y decidir la clase en función del signo
        return np.sign(y_pred)
        
def train_adaboost_for_digit(digit, T, A, verbose=False):
    # Cargar los datos de MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Preprocesar los datos - aplanar y normalizar
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255.
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255.
    
    # Filtrar los datos para el dígito específico vs todos los demás
    y_train_digit = np.where(y_train == digit, 1, -1)
    y_test_digit = np.where(y_test == digit, 1, -1)
    
    # Iniciar el clasificador AdaBoost
    clf = Adaboost(T=T, A=A)
    
    # Entrenar el clasificador y medir el tiempo de entrenamiento
    start_time = time.time()
    clf.fit(X_train, y_train_digit, verbose=verbose)
    end_time = time.time()
    
    # Calcular las tasas de acierto
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train_digit, y_train_pred)
    test_accuracy = accuracy_score(y_test_digit, y_test_pred)
    
    # Imprimir las tasas de acierto
    print(f"Entrenando clasificador Adaboost para el dígito {digit}, T={T}, A={A}")
    if verbose:
        for i, stump in enumerate(clf.lista_clasificadores):
            print(f"Añadido clasificador {i+1}: {stump.caracteristica_index}, {stump.umbral:.4f}, "
                f"{'+' if stump.polaridad == 1 else '-'}, {stump.alpha:.6f}")
    print(f"Tasas acierto (train, test) y tiempo: {train_accuracy*100:.2f}%, {test_accuracy*100:.2f}%, {end_time - start_time:.3f} s")


train_adaboost_for_digit(digit=9, T=20, A=10, verbose=True)
        


import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class DecisionStump:
    def __init__(self, n_caracteristicas):
        self.caracteristica_index = np.random.randint(0, n_caracteristicas) # Seleccionar una característica al azar. n_caracteristicas es el número total de características disponibles.
        self.umbral = np.random.uniform() # Inicializar el umbral.
        self.polaridad = np.random.choice([1, -1]) # Polaridad puede ser 1 o -1, se elige al azar.
        self.alpha = None # Valor de alpha (peso del clasificador), inicializado a None. Este será calculado durante el entrenamiento.
    
    def predict(self, X):
        n_samples = X.shape[0]
        predicciones = np.ones(n_samples)
        feature_column = X[:, self.caracteristica_index]
        if self.polaridad == 1:
            predicciones[feature_column < self.umbral] = -1
        else:
            predicciones[feature_column >= self.umbral] = -1
        return predicciones


# class Adaboost:
#     def __init__(self, T=5, A=20):
#         self.T = T
#         self.A = A
#         self.lista_clasificadores = []

#     def fit(self, X, Y, verbose=False):
#         n_samples, n_caracteristicas = X.shape
#         D = np.full(n_samples, (1 / n_samples))

#         for t in range(self.T):
#             clf_best = None
#             min_error = float('inf')

#             for a in range(self.A):
#                 clf = DecisionStump(n_caracteristicas)

#                 predicciones = clf.predict(X)
#                 error = np.sum(D[Y != predicciones])

#                 if error < min_error:
#                     min_error = error
#                     clf_best = clf

#             # Calculate alpha
#             clf_best.alpha = 0.5 * np.log((1.0 - min_error) / (min_error + 1e-10))

#             # Update weights
#             D *= np.exp(-clf_best.alpha * Y * clf_best.predict(X))
#             D /= np.sum(D)

#             self.lista_clasificadores.append(clf_best)

#             if verbose:
#                 print(f'Ronda {t+1}, Caracteristica {clf_best.caracteristica_index}, '
#                     f'Umbral {clf_best.umbral}, Polaridad {clf_best.polaridad}, '
#                     f'Error {min_error}, Alpha {clf_best.alpha}')

#     def predict(self, X):
#         clf_preds = [clf.alpha * clf.predict(X) for clf in self.lista_clasificadores]
#         y_pred = np.sum(clf_preds, axis=0)
#         return np.sign(y_pred)
class Adaboost:
    def __init__(self, T=5, A=20):
        self.T = T
        self.A = A
        self.umbral_mejora = 0.01
        self.incremento_A = 5
        self.min_iteraciones = 10
        self.early_stopping = True
        self.lista_clasificadores = []

    def fit(self, X, Y, mejora1E, verbose=False):
        n_samples, n_caracteristicas = X.shape
        D = np.full(n_samples, (1 / n_samples))

        clfmejorado = Adaboost(T=self.T, A=self.A)

        last_error = float('inf')
        for t in range(self.T):
            clf_best = None
            min_error = float('inf')

            for a in range(self.A):
                clf = DecisionStump(n_caracteristicas)
                predicciones = clf.predict(X)
                error = np.sum(D[Y != predicciones])

                if error < min_error:
                    min_error = error
                    clf_best = clf

            clf_best.alpha = 0.5 * np.log((1.0 - min_error) / (min_error + 1e-10))
            D *= np.exp(-clf_best.alpha * Y * clf_best.predict(X))
            D /= np.sum(D)

            self.lista_clasificadores.append(clf_best)

            if mejora1E: # SI se activa la booleana se hará el entrenamiento con la mejora de la 1E
                clfimejorado, parar = tarea1E(self, last_error, min_error, t)
                self.A = clfmejorado.A
                if parar:
                    break
                else:
                    last_error = min_error

            if verbose:
                print(f'Ronda {t+1}, Error {min_error}, Alpha {clf_best.alpha}')

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.lista_clasificadores]
        y_pred = np.sum(clf_preds, axis=0)
        return np.sign(y_pred)

def tarea1E(clf, last_error, min_error, t):
    parar = False
    # Ajuste dinámico de A y detención temprana Tarea 1E
    if last_error - min_error < clf.umbral_mejora:
        clf.A += clf.incremento_A
        if clf.early_stopping and t > clf.min_iteraciones:
            parar = True

    return clf, parar
################################################################ Tarea 1B #########################################################################

def tarea1B(digito, T, A, verbose=False):
    # Cargar los datos de MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Preprocesar los datos - aplanar y normalizar
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

    # Filtrar los datos para el dígito específico vs todos los demás
    y_train_digito = np.where(y_train == digito, 1, -1)
    y_test_digito = np.where(y_test == digito, 1, -1)
    
    # Iniciar el clasificador AdaBoost
    clf = Adaboost(T=T, A=A)
    
    # Entrenar el clasificador y medir el tiempo de entrenamiento
    start_time = time.time()
    clf.fit(X_train, y_train_digito, mejora1E=True, verbose=verbose)
    end_time = time.time()

    # Calcular las tasas de acierto
    y_train_pred = clf.predict(X_train) #ESTE PREDICT ES EL DE ADABOOST
    y_test_pred = clf.predict(X_test)   # Y ESTE PREDICT TB ES EL DE ADABOOST
    train_accuracy = accuracy_score(y_train_digito, y_train_pred)
    test_accuracy = accuracy_score(y_test_digito, y_test_pred)
    
    # Imprimir las tasas de acierto
    print(f"Entrenando clasificador Adaboost para el dígito {digito}, T={T}, A={A}")
    if verbose:
        for i, c in enumerate(clf.lista_clasificadores):
            print(f"Añadido clasificador {i+1}: {c.caracteristica_index}, {c.umbral:.4f}, "f"{'+' if c.polaridad == 1 else '-'}, {c.alpha:.6f}")

    print(f"Tasas acierto (train, test) y tiempo: {train_accuracy*100:.2f}%, {test_accuracy*100:.2f}%, {end_time - start_time:.3f} s")

    return train_accuracy, test_accuracy, end_time - start_time

################################################################ Tarea 1C #########################################################################


def tarea1C(digito, train_accuracies_aux, test_accuracies_aux, training_times_aux):
    #Define los rangos de valores para T y A que deseas probar
    T_values = [10, 20, 30, 40]
    A_values = [5, 10, 20, 30]

    #Tarea 1C recorre T y A y genera graficas
    for T, A in zip(T_values, A_values):
        train_accuracy, test_accuracy, training_time = tarea1B(digito, T, A)
        train_accuracies_aux.append(train_accuracy)
        test_accuracies_aux.append(test_accuracy)
        training_times_aux.append(training_time)

    # Asegúrate de que TA_producto tenga la misma longitud que las otras listas
    TA_producto = [T * A for T, A in zip(T_values, A_values)]
    
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('T x A')
    ax1.set_ylabel('Accuracy', color='tab:red')
    ax1.plot(TA_producto, train_accuracies_aux, 'o-', color='tab:red', label='Train Accuracy')
    ax1.plot(TA_producto, test_accuracies_aux, 'o-', color='tab:orange', label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Time (seconds)', color='tab:blue')
    ax2.plot(TA_producto, training_times_aux, 's-', color='tab:blue', label='Time to Train')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title('Accuracy and Training Time vs T x A')
    plt.show()
    plt.close()

def tarea1D():
    digitos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for digito in digitos: # Tarea 1D para cada dígito se entrena un clasificador Adaboost
        train_accuracies_aux = []
        test_accuracies_aux = []
        training_times_aux = []
        print("DIGITO: ", digito)

        # Generar y mostrar gráficas después de procesar todos los valores de T y A para un dígito
        tarea1C(digito, train_accuracies_aux, test_accuracies_aux, training_times_aux)


def tarea2A():
    print()
    print("Tarea 2A")

    # Cargar el conjunto de datos MNIST
    mnist = fetch_openml('mnist_784', parser='auto')
    X, y = mnist["data"], mnist["target"]

    # Convertir las etiquetas a enteros
    y = y.astype(int)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el clasificador AdaBoost
    ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)

    # Entrenar el modelo
    ada_clf.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = ada_clf.predict(X_test)

    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_pred)
    # Resultados
    print(f'La precisión del modelo es del {round(accuracy_score(y_test, y_pred) * 100, 2)}%')
    print(f'El informe de clasificación es: \n {classification_report(y_test, y_pred)}')




def main():
    
    train_accuracy, test_accuracy, training_time = tarea1B(9, 10, 20, True)

    tarea2A()

    
    




if __name__ == "__main__":
    main()
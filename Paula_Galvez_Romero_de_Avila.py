import matplotlib.pyplot as plt
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
            predicciones[X[:, self.caracteristica_index] >= self.umbral] = 1 # Si la característica es mayor que el umbral, la predicción es -1.
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
        print("Obtener el número de observaciones y de características por observación de X")
        n_samples, n_caracteristicas = X.shape 
                                                # n_samples número total de imágenes en el conjunto de datos
                                                # n_caracterisitcas n pixeles(cada píxel es una característica) de cada imagen
        print("Iniciar pesos de las observaciones a 1/n_samples")
        w = np.full(n_samples, (1 / n_samples)) 
        #w : array de pesos. Cada elemento de w representa el peso inicial de cada imagen en tu conjunto de datos.
        print("n_samples: ", n_samples, " y n_caracteristicas: ", n_caracteristicas)
        print("w", w)

        # Bucle de entrenamiento Adaboost # Bucle de entrenamiento Adaboost: desde 1 hasta T repetir
        print()
        for t in range(1, self.T): ## Hasta T clasificadores
            print("CLASIFICADOR T: ", t)
            print()
            print("Iniciar el clasificador con el mayor error posible para asegurar que se actualice en el primer intento")
            clf = DecisionStump(n_caracteristicas)
            min_error = float('inf')
            print(clf, min_error)

            # Buscar el mejor clasificador débil: desde 1 hasta A repetir
            # Calcular predicciones de ese clasificador para todas las observaciones
            for a in range(1, self.A):
                print("INTENTO A: ", a)
                print("Generar un umbral y característica aleatorios para el clasificador")
                caracteristica_index = np.random.randint(0, n_caracteristicas) # Crear un nuevo clasificador débil aleatorio
                umbral = np.random.uniform(np.min(X[:, caracteristica_index]), np.max(X[:, caracteristica_index])) 
                polaridad = np.random.choice([1, -1])
                print("caracteristica_index: ",caracteristica_index)
                print("umbral: ", umbral)
                print("polaridad: ", polaridad)

                print("Actualizar el clasificador débil con estos parámetros aleatorios")
                clf.caracteristica_index = caracteristica_index
                clf.umbral = umbral
                clf.polaridad = polaridad

                print()
                print("Realizamos predicciones con el clasificador débil actual")
                predicciones = clf.predict(X)
                print("predicciones: ", predicciones)
                
                print()
                print("Calcular el error ponderado de las predicciones comparar predicciones con los valores deseados") 
                print(" Y acumular los pesos de las observaciones mal clasificadas")
                error = np.sum(w[Y != predicciones])
                print("error: ", error)
                print("min_error: ", min_error)

                print()
                print("Actualizamos mejor clasificador hasta el momento: el que tenga menor error")
                # Si el error es menor que el error mínimo, actualizar el clasificador y el error mínimo
                if error < min_error:
                    print("error es menor a min_error -> SE ACTUALIZA")
                    min_error = error
                    best_clf = clf
                    print("Nuevo min_error: ", min_error)
                    print("NUEVO best_clf: ", best_clf)
                    print("Actualizar el umbral y la polaridad del mejor clf ")
                    best_clf.umbral = np.random.uniform(np.min(X[:, clf.caracteristica_index]), np.max(X[:, clf.caracteristica_index]))
                    best_clf.polaridad = np.random.choice([1, -1])
                    print("best_clf.umbral: ", best_clf.umbral, " best_clf.polaridad: ", best_clf.polaridad)
                else:
                    print("error es mayor a min_error -> NO SE ACTUALIZA")

            # if min_error == 0:
            #     min_error = 1e-10  # Para evitar división por cero en el cálculo de alpha

            print()
            print("Calculamos el valor de alfa y las predicciones del mejor clasificador débil")
            #best_clf.alpha = 0.5 * np.log((1 - min_error) / min_error)
            best_clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))
            print("best_clf.alpha: ", best_clf.alpha)

            print()
            print("Guardamos el mejor clasificador en la lista de clasificadores de Adaboost")
            self.lista_clasificadores.append(best_clf)
            print("lista_clasificadores: ", self.lista_clasificadores)
            
            print()
            print("Actualizamos los pesos de las observaciones en función de las predicciones, los valores deseados y alfa")
            print("Y Normalizamos a 1 los pesos")
            predicciones = best_clf.predict(X)
            print("predicciones: ", predicciones)
            w *= np.exp(-best_clf.alpha * Y * predicciones)
            w /= np.sum(w)
            print("w: ", w)
            print()

            if verbose:
                print(f'Ronda {t+1}, Caracteristica {best_clf.caracteristica_index}, Umbral {best_clf.umbral}, Polaridad {best_clf.polaridad}, Error {min_error}, Alpha {best_clf.alpha}')
            
            print()

    # Método para obtener una predicción con el clasificador fuerte Adaboost 
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.lista_clasificadores] # Calcular las predicciones de cada clasificador débil para cada input multiplicadas por su alfa
        y_pred = np.sum(clf_preds, axis=0) # Sumar para cada input todas las predicciones ponderadas y decidir la clase en función del signo
        return np.sign(y_pred)

################################################################ Tarea 1B #########################################################################

def entrenamiento(digito, T, A, verbose=False):
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
    clf.fit(X_train, y_train_digito, verbose=verbose)
    end_time = time.time()

    # Calcular las tasas de acierto
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train_digito, y_train_pred)
    test_accuracy = accuracy_score(y_test_digito, y_test_pred)
    #accuracy es precision / exactitud
    
    # Imprimir las tasas de acierto
    print(f"Entrenando clasificador Adaboost para el dígito {digito}, T={T}, A={A}")
    if verbose:
        for i, c in enumerate(clf.lista_clasificadores):
            print(f"Añadido clasificador {i+1}: {c.caracteristica_index}, {c.umbral:.4f}, "f"{'+' if c.polaridad == 1 else '-'}, {c.alpha:.6f}")

    print(f"Tasas acierto (train, test) y tiempo: {train_accuracy*100:.2f}%, {test_accuracy*100:.2f}%, {end_time - start_time:.3f} s")

    # Devolver precisión y tiempo
    return train_accuracy, test_accuracy, end_time - start_time

################################################################ Tarea 1C #########################################################################


def generarGraficasPlot(T_values, A_values, train_accuracies, test_accuracies, training_times):
    
    # train_accuracies = []
    # test_accuracies = []
    # times = []
    TA_producto = [T * A for T in T_values for A in A_values]

    # # Prueba con diferentes valores de T y A y guarda los resultados
    # for T in T_values:
    #     for A in A_values:
    #         train_accuracy, test_accuracy, training_time = entrenamiento(digito, T, A, True) # Si decido descomentar esta linea hay que añadir por parametro a digito
    #         train_accuracies.append(train_accuracy)
    #         test_accuracies.append(test_accuracy)
    #         times.append(training_time)

    # Ahora, crea la gráfica de doble eje
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('T x A')
    ax1.set_ylabel('Accuracy', color='tab:red')
    ax1.plot(TA_producto, train_accuracies, 'o-', color='tab:red', label='Train Accuracy')
    ax1.plot(TA_producto, test_accuracies, 'o-', color='tab:orange', label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Time (seconds)', color='tab:blue')
    ax2.plot(TA_producto, training_times, 's-', color='tab:blue', label='Time to Train')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title('Accuracy and Training Time vs T x A')
    plt.show()
    plt.close()  # Cierra la ventana de la gráfica

def main():
    # Define los rangos de valores para T y A que deseas probar
    T_values = [10, 5, 10]# 10, 20, 5, 30]
    A_values = [20, 10, 30]# 40, 40, 30, 30] 
    
    digitos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for digito in digitos: # Tarea 1D
        train_accuracies_matrizAux = []
        test_accuracies_matrizAux = []
        training_times_matrizAux = []
        print("DIGITO: ", digito)
        for T in T_values:
            for A in A_values:
                print()
                print("Clasificador T:", T, " intento : ", A)
                print()
                train_accuracy, test_accuracy, training_time = entrenamiento(digito, T, A)
                print()
                print("train_accuracy: ", train_accuracy)
                print("test_accuracy: ", test_accuracy)
                print("training_time: ", training_time)
                train_accuracies_matrizAux.append(train_accuracy)
                test_accuracies_matrizAux.append(test_accuracy)
                training_times_matrizAux.append(training_time)
                print()
                print("train_accuracies_matrizAux ", train_accuracies_matrizAux)
                print("test_accuracies_matrizAux ", test_accuracies_matrizAux)
                print("training_times_matrizAux ", training_times_matrizAux)
        print()
        print("Fin del entrenamiento para el dígito ", digito)
        generarGraficasPlot(T_values, A_values, train_accuracies_matrizAux, test_accuracies_matrizAux, training_times_matrizAux)
        print("#### GRAFICA ####")
        print()


if __name__ == "__main__":
    main()

    
    

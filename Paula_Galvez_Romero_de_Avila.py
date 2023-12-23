import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE


from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# Cargar los datos de MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocesar los datos - aplanar y normalizar
X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

class DecisionStump:
    def __init__(self, n_caracteristicas):
        self.caracteristica = np.random.randint(0, n_caracteristicas) # Seleccionar una característica al azar. n_caracteristicas es el número total de características disponibles.
        self.umbral = np.random.uniform() # Inicializar el umbral.
        self.polaridad = np.random.choice([1, -1]) # Polaridad puede ser 1 o -1, se elige al azar.
        self.alpha = None # Valor de alpha (peso del clasificador), inicializado a None. Este será calculado durante el entrenamiento.
    
    def predict(self, X):
        n_samples = X.shape[0]
        predicciones = np.ones(n_samples)
        feature_column = X[:, self.caracteristica]
        if self.polaridad == 1:
            predicciones[feature_column < self.umbral] = -1
        else:
            predicciones[feature_column >= self.umbral] = -1
        return predicciones

class Adaboost:
    def __init__(self, T=5, A=20):
        self.T = T
        self.A = A
        self.lista_clasificadores = []
        

    def fit(self, X, Y, verbose=False):
        n_samples, n_caracteristicas = X.shape
        D = np.full(n_samples, (1 / n_samples))
        #D = sample_weight.copy()  # Inicializar con los pesos de las muestras
        #clfmejorado = Adaboost(T=self.T, A=self.A)

        #last_error = float('inf')
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

            # if mejora1E: # SI se activa la booleana se hará el entrenamiento con la mejora de la 1E
            #     clfimejorado, parar = tarea1E(self, last_error, min_error, t)
            #     self.A = clfmejorado.A
            #     if parar:
            #         break
            #     else:
            #         last_error = min_error

            if verbose:
                print(f'Ronda {t+1}, Error {min_error}, Alpha {clf_best.alpha}')

    def predict(self, X, binary_output=False):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.lista_clasificadores]
        y_pred = np.sum(clf_preds, axis=0)
        #return np.sign(y_pred)
        #return y_pred
        return y_pred if binary_output else np.sign(y_pred)

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
    #global X_train, y_train, X_test, y_test
    # Cargar los datos de MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocesar los datos - aplanar y normalizar
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

    # Filtrar los datos para el dígito específico vs todos los demás
    y_train_digito = np.where(y_train == digito, 1, -1)
    y_test_digito = np.where(y_test == digito, 1, -1)

    # Aplicar SMOTE para balancear las clases
    # smote = SMOTE()
    # X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_digito)
    
    #Equilibrar el conjunto de entrenamiento
    # Supongamos que usamos submuestreo para la clase mayoritaria
    clase_positiva = X_train[y_train_digito == 1]
    clase_negativa = X_train[y_train_digito == -1]

    # Submuestrear la clase negativa para igualar el número de muestras
    clase_negativa_sub = clase_negativa[:clase_positiva.shape[0]]
    X_train_eq = np.concatenate([clase_positiva, clase_negativa_sub])
    y_train_eq = np.concatenate([np.ones(clase_positiva.shape[0]), -np.ones(clase_positiva.shape[0])])

    # Iniciar el clasificador AdaBoost
    clf = Adaboost(T=T, A=A)
    
    # Entrenar el clasificador y medir el tiempo de entrenamiento
    start_time = time.time()
    clf.fit(X_train_eq, y_train_eq, verbose=verbose)
    end_time = time.time()

    # Calcular las tasas de acierto
    y_train_pred = clf.predict(X_train_eq, binary_output=False) #ESTE PREDICT ES EL DE ADABOOST
    y_test_pred = clf.predict(X_test, binary_output=False)   # Y ESTE PREDICT TB ES EL DE ADABOOST
    train_accuracy = accuracy_score(y_train_eq, y_train_pred)
    test_accuracy = accuracy_score(y_test_digito, y_test_pred)
    
    # Imprimir las tasas de acierto
    print(f"Entrenando clasificador Adaboost para el dígito {digito}, T={T}, A={A}")
    if verbose:
        for i, c in enumerate(clf.lista_clasificadores):
            print(f"Añadido clasificador {i+1}: {c.caracteristica}, {c.umbral:.4f}, "f"{'+' if c.polaridad == 1 else '-'}, {c.alpha:.6f}")

    print(f"Tasas acierto (train, test) y tiempo: {train_accuracy*100:.2f}%, {test_accuracy*100:.2f}%, {end_time - start_time:.3f} s")

    return train_accuracy, test_accuracy, end_time - start_time, clf, y_test_pred

# def tarea1B(digito, T, A, verbose=False):
#     global X_train, y_train, X_test, y_test

#     # Filtrar los datos para el dígito específico vs todos los demás
#     y_train_digito = np.where(y_train == digito, 1, -1)
#     y_test_digito = np.where(y_test == digito, 1, -1)

#     # Equilibrar el conjunto de entrenamiento
#     # # Supongamos que usamos submuestreo para la clase mayoritaria
#     # clase_positiva = X_train[y_train_digito == 1]
#     # clase_negativa = X_train[y_train_digito == -1]

#     # # Submuestrear la clase negativa para igualar el número de muestras
#     # clase_negativa_sub = clase_negativa[:clase_positiva.shape[0]]
#     # X_train_eq = np.concatenate([clase_positiva, clase_negativa_sub])
#     # y_train_eq = np.concatenate([np.ones(clase_positiva.shape[0]), -np.ones(clase_positiva.shape[0])])

#     # # Asignar pesos para equilibrar el peso de los dígitos y los no-dígitos
#     # n_digito = np.sum(y_train_eq == 1)
#     # n_no_digito = np.sum(y_train_eq == -1)
#     # total_muestras = n_digito + n_no_digito
#     # peso_digito = (total_muestras / (2 * n_digito))
#     # peso_no_digito = (total_muestras / (2 * n_no_digito))
#     # pesos = np.where(y_train_eq == 1, peso_digito, peso_no_digito)

#     # Iniciar el clasificador AdaBoost
#     clf = Adaboost(T=T, A=A)
#     # Entrenar el clasificador y medir el tiempo de entrenamiento
#     start_time = time.time()
#     clf.fit(X_train, y_train_digito, verbose=verbose)
#     #clf.fit(X_train_eq, y_train_eq, sample_weight=pesos, verbose=verbose)
#     end_time = time.time()

#     # Resto del código para realizar predicciones y calcular la precisión
#     y_train_pred = clf.predict(X_train, binary_output=False)
#     y_test_pred = clf.predict(X_test, binary_output=False)
#     train_accuracy = accuracy_score(y_train, y_train_pred)
#     test_accuracy = accuracy_score(y_test_digito, y_test_pred)

#     # Imprimir las tasas de acierto
#     print(f"Entrenando clasificador Adaboost para el dígito {digito}, T={T}, A={A}")
#     if verbose:
#         for i, c in enumerate(clf.lista_clasificadores):
#             print(f"Añadido clasificador {i+1}: {c.caracteristica}, {c.umbral:.4f}, "f"{'+' if c.polaridad == 1 else '-'}, {c.alpha:.6f}")

#     print(f"Tasas acierto (train, test) y tiempo: {train_accuracy*100:.2f}%, {test_accuracy*100:.2f}%, {end_time - start_time:.3f} s")

#     return train_accuracy, test_accuracy, end_time - start_time, clf

################################################################ Tarea 1C #########################################################################


def tarea1D(): # entrenar el adaboost multiclase
    clasificadores = []
    digitos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    for i, digito in enumerate(digitos):
        clf, y_testpred = tarea1C(i, digito, axs)
        clasificadores.append((clf, y_testpred ))

    plt.tight_layout()
    plt.show()

    y_pred =  predecir_multiclase(X_test, clasificadores)
    return y_pred

def predecir_multiclase(X, clasificadores):
    # Inicializar la matriz de predicciones
    predicciones_totales = np.zeros((X.shape[0], len(clasificadores)))

    # Calcular las predicciones para cada clasificador
    for idx, clf in enumerate(clasificadores):
        predicciones = clf[0].predict(X, binary_output=False)
        #predicciones = clf[1]
        predicciones_totales[:, idx] = predicciones

    # Elegir la clase con la mayor puntuación para cada muestra
    y_pred = np.argmax(predicciones_totales, axis=1)

    accuracy = accuracy_score(y_test, y_pred) # Evaluar las predicciones con respecto a las etiquetas reales
    print(f"Precisión del clasificador multiclase: {accuracy * 100:.2f}%")
    return predicciones_totales



def tarea1C(i, digito, axs): # entrenamiento y gráficas
    #global X_train, y_train, X_test, y_test

    T_values = [10, 20, 30, 40]
    A_values = [5, 10, 20, 30]
    #clasificadores = []

    train_accuracies_aux = []
    test_accuracies_aux = []
    training_times_aux = []
    
    for T, A in zip(T_values, A_values):
        train_accuracy, test_accuracy, training_time, clf, y_testpred = tarea1B(digito, T, A)
        train_accuracies_aux.append(train_accuracy)
        test_accuracies_aux.append(test_accuracy)
        training_times_aux.append(training_time)
        #clasificadores.append(clf)

    print()
    mediatrain = np.mean(train_accuracies_aux)
    mediatest = np.mean(test_accuracies_aux)
    mediatimes = np.mean(training_times_aux)
    print(f"Para el digito {digito}")
    print(f"El modelo tiene una precision de entramiento {mediatrain}")
    print(f"El modelo tiene una precision de test {mediatest}")
    print(f"En el tiempo medio de {mediatimes}")
    print()

    TA_producto = [T * A for T, A in zip(T_values, A_values)]

    # Aquí utilizamos los ejes proporcionados
    ax = axs[i]
    ax.set_title(f'Digito: {digito}')
    ax.set_xlabel('T x A')
    ax.set_ylabel('Accuracy', color='tab:red')
    ax.plot(TA_producto, train_accuracies_aux, 'o-', color='tab:red', label='Train Accuracy')
    ax.plot(TA_producto, test_accuracies_aux, 'o-', color='tab:orange', label='Test Accuracy')
    ax.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax.twinx()
    ax2.set_ylabel('Time (seconds)', color='tab:blue')
    ax2.plot(TA_producto, training_times_aux, 's-', color='tab:blue', label='Time to Train')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    return clf, y_testpred


def tarea2A(n): # Clasificador de scikit-learn

    # Cargar el conjunto de datos MNIST
    mnist = fetch_openml('mnist_784', parser='auto')
    X, y = mnist["data"], mnist["target"]

    # Convertir las etiquetas a enteros
    y = y.astype(int)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el clasificador AdaBoost
    #ada_clf = AdaBoostClassifier(n_estimators=n, random_state=42)
    ada_clf = AdaBoostClassifier(
        n_estimators=n, 
        learning_rate=1.0, # Puedes ajustar esto según sea necesario
        algorithm='SAMME.R', # Usa SAMME.R que suele ser más eficaz
        random_state=42
    )

    start_time = time.time()
    # Entrenar el modelo
    ada_clf.fit(X_train, y_train)
    end_time = time.time()
    
    # Realizar predicciones
    y_pred = ada_clf.predict(X_test)

    # Resultados
    print(f'La precisión del modelo es del {round(accuracy_score(y_test, y_pred) * 100, 2)}%')
    print(f'El informe de clasificación es: \n {classification_report(y_test, y_pred)}')
    print(f"Tiempo: {end_time - start_time:.3f} s")

    return y_pred



def tarea2B(y_preds1D, y_preds2A):
    print("Comparativa resultados entre el Adaboost Binario y el Adaboostclassifier")
    T_values = [10, 20, 30, 40]
    # Asumimos que y_preds_mi_adaboost y y_preds_sklearn son listas de y_pred para cada T
    aciertos_mi_adaboost = [accuracy_score(y_test, y_pred) for y_pred in y_preds1D]
    aciertos_sklearn = [accuracy_score(y_test, y_pred) for y_pred in y_preds2A]
    
    plt.figure(figsize=(10, 6))
    plt.plot(T_values, aciertos_mi_adaboost, marker='o', linestyle='-', color='blue', label='Mi Adaboost')
    plt.plot(T_values, aciertos_sklearn, marker='x', linestyle='--', color='red', label='AdaboostClassifier Scikit-learn')
    
    plt.title('Comparación de Aciertos entre Mi Adaboost y AdaboostClassifier')
    plt.xlabel('T (Número de Iteraciones)')
    plt.ylabel('Porcentaje de Acierto')
    plt.xticks(T_values)
    plt.legend()
    plt.grid(True)
    plt.show()

    tarea1D()

def tarea2C(): # Adaboostclassifier con el DecisioTree

    # Cargar el conjunto de datos MNIST
    mnist = fetch_openml('mnist_784', parser='auto')
    X, y = mnist["data"], mnist["target"]

    # Convertir las etiquetas a enteros
    y = y.astype(int)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el clasificador de árbol de decisión
    # dt_clf = DecisionTreeClassifier(max_depth=1)
    dt_clf = DecisionTreeClassifier(
        criterion='gini', # Uso de entropía para una mejor división
        max_depth=5,        # Profundidad máxima
        min_samples_split=2, # Mínimo de muestras para dividir
        min_samples_leaf=1,  # Mínimo de muestras en nodo hoja
        random_state=42
    )

    # Crear el clasificador AdaBoost
    #ada_clf = AdaBoostClassifier(estimator=dt_clf, n_estimators=50, random_state=42)
    ada_clf = AdaBoostClassifier(
        estimator = dt_clf,
        n_estimators=50, 
        learning_rate=1.0, # Puedes ajustar esto según sea necesario
        algorithm='SAMME.R', # Usa SAMME.R que suele ser más eficaz
        random_state=42
    )

    start_time = time.time()
    # Entrenar el modelo
    ada_clf.fit(X_train, y_train)
    end_time = time.time()
    
    # Realizar predicciones
    y_pred = ada_clf.predict(X_test)

    # Resultados
    print(f'La precisión del modelo es del {round(accuracy_score(y_test, y_pred) * 100, 2)}%')
    print(f'El informe de clasificación es: \n {classification_report(y_test, y_pred)}')
    print(f"Tiempo: {end_time - start_time:.3f} s")

    return y_pred
    

def tarea2D(): # MLP
    # Cargar el conjunto de datos MNIST
    (X, y), (X_test, y_test) = mnist.load_data()

    # Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar los datos (si no se ha hecho previamente)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Asegurarse de que los datos están en el formato correcto
    X_train = X_train.reshape((-1, 784))
    X_test = X_test.reshape((-1, 784))

    # Crear el modelo MLP
    model = Sequential()
    model.add(Dense(256, input_shape=(784,), activation='relu'))  # Capa oculta
    model.add(Dense(10, activation='softmax'))  # Capa de salida

    # Compilar el modelo
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(f'Test Accuracy: {test_acc}')
    print(f'Test Loss: {test_loss}')

    # Opcional: Devolver el modelo y el historial para análisis posterior
    return model, history

def tarea2E(): # CNN
    # Cargar el conjunto de datos MNIST
    (X, y), (X_test, y_test) = mnist.load_data()

    # Normalizar y redimensionar los datos
    X = X.reshape((-1, 28, 28, 1)).astype('float32') / 255
    X_test = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255

    # Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construir el modelo CNN
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compilar el modelo
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(f'Test Accuracy: {test_acc}')
    print(f'Test Loss: {test_loss}')

    # Opcional: Devolver el modelo y el historial para análisis posterior
    return model, history


def main():
    #global X_train, y_train, X_test, y_test
    #print("########## Tarea 1B: entrenamiento básico con el dígito 9 ADABOOST BINARIO #########")
    #train_accuracy, test_accuracy, training_time = tarea1B(9, 10, 20, True)

    # La 1E, ya va implementada
    # Si llamo a tarea 1D que es el adaboost multiclase, para cada digito llamará a la tarea 1C, donde se llamara a la tarea 1B y se entrenara cada digito
    print()
    print("Tarea 1D : Clasificador multiclase")
    predicts1D = tarea1D()

    print()
    print("Tarea 2A: Adaboostclassifier de scikit-learn")
    # T_values = [10, 20, 30, 40]
    # predicts2A = []
    # for t in T_values:
    #     y_pred2A = tarea2A(t)
    #     predicts2A.append(accuracy_score(y_test, y_pred2A))
    
    #y_pred2A = tarea2A(50)

    print()
    print("Tarea 2B: Comparando el adaboost classifier con el binario")
    #tarea2B(predicts1D, predicts2A)
    
    print()
    print("Tarea 2C: Adaboostclassifier con DecisionTree")
    #y_pred2C = tarea2C()

    print()
    print("Volvemos a compararlo(Tarea2B) con el binario")
    #tarea2B(predicts1D, y_pred2C)

    print()
    print("Tarea 2D: Perceptron multicapa (MLP)")
    #tarea2D()

    print()
    print("Tarea 2E: Red neuronal convuncional (CNN)")
    #tarea2E()

    print()
    print("Tarea 2F: COMPARANDO ADABOOST MULTICLASE - ADABOOSTCLASSIFIER DE SIKIT-LEARN - MLP - CNN")
    #tarea2F()
    print()


    # Dividir X, y en conjuntos de entrenamiento y validación
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    

if __name__ == "__main__":
    main()
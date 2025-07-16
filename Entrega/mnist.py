# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 23:02:36 2025

@author: rafae
"""

# librerías usadas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns
# %%

# Dataframe usado

mnist = pd.read_csv("mnist_c_fog_tp.csv")

# %%
#                                              CONSIGNA 1

# %%

# análisis de las columnas, filas y valores del dataframe

print(f'El dataframe mnist tiene {len(mnist)} filas y {len(mnist.loc[0])} columnas')
print(f'El dataframe mnist tiene {len(mnist)*len(mnist.loc[0])} datos')
print(f"El valor minimo de cada pixel es: {mnist.drop(['labels', 'Unnamed: 0'], axis=1).min().min()} \n el máximo es: {mnist.drop(['labels', 'Unnamed: 0'], axis=1).max().max()}")

# %%

#se elimina la columna Unnamed: 0 correspondiente al indice

mnist = mnist.drop('Unnamed: 0', axis=1)

# %%

# cramos un dataframe sin el atributo label para poder manipular las imagenes

mnist_manipulable =  mnist.drop('labels', axis=1)

# %%

# se normaliza la base de datos

mnist_manipulable = mnist_manipulable.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# %% a.
# %%



# creamos una función para graficar los números

def graficar_imagen(numero, titulo, ax):
    img = np.array(numero).reshape((28,28))
    ax.imshow(img, cmap='gray')
    ax.set_title(titulo)
    ax.axis('off')
    
# %%

# graficamos dos números aleatorios

numero_random1 = random.randint(0,69999)
numero_random2 = random.randint(0,69999)

fig, ax = plt.subplots(1,2)
graficar_imagen(mnist_manipulable.loc[numero_random1], 'imagen aleatoria' , ax[0])
graficar_imagen(mnist_manipulable.loc[numero_random2], 'imagen aleatoria' , ax[1])

# %%

 # graficamos números especificos elegidos aleatoriamente
  
fig, ax = plt.subplots(1,2)
graficar_imagen(mnist_manipulable.loc[5062], 'imagen aleatoria' , ax[0])
graficar_imagen(mnist_manipulable.loc[44189], 'imagen aleatoria' , ax[1])

# %%


 
fig, ax = plt.subplots(1,2)
graficar_imagen(mnist_manipulable.loc[5062], 'imagen aleatoria' , ax[0])
graficar_imagen(mnist_manipulable.loc[44189], 'imagen aleatoria' , ax[1])


cuadrado1 = patches.Rectangle((0, 0), 27, 27, linewidth=1, edgecolor='red', facecolor='none')
cuadrado2 = patches.Rectangle((3, 5), 20, 21, linewidth=1, edgecolor='red', facecolor='none')

ax[0].add_patch(cuadrado1)
ax[0].add_patch(cuadrado2)

cuadrado1 = patches.Rectangle((0, 0), 27, 27, linewidth=1, edgecolor='red', facecolor='none')
cuadrado2 = patches.Rectangle((5, 1), 19, 21, linewidth=1, edgecolor='red', facecolor='none')

ax[1].add_patch(cuadrado1)
ax[1].add_patch(cuadrado2)
# %%

# función para generar una base de datos con el promedio normalizado de todas 
# las imagenes

def imagen_promedio(n):
    numero = mnist.query(f'labels == {n}')
    numero_sin_label = numero.drop(columns=['labels'])
    numero_mean = numero_sin_label.mean()
    numero_normalizado = numero_mean.apply(lambda x: (x - numero_mean.min()) / (numero_mean.max() - numero_mean.min()))
    return numero_normalizado

# %%

# agrupamos imagenes promedio de 1 al 9

fig, ax = plt.subplots(2, 5, figsize=(10,4.5))

columna = 0
fila = 0
for i in range(10):
    if i == 5:
        fila = 1
        columna = 0
        
    graficar_imagen(imagen_promedio(i), f'imagen \n promedio {i}', ax[fila,columna])
    columna += 1 
# %%

# agrupamos imagenes promedio de 1 al 9 con rectangulos

fig, ax = plt.subplots(2, 5, figsize=(10,4.5))

columna = 0
fila = 0
for i in range(10):
    if i == 5:
        fila = 1
        columna = 0
        
    graficar_imagen(imagen_promedio(i), f'imagen \n promedio {i}', ax[fila,columna])
    
    cuadrado1 = patches.Rectangle((0, 0), 27, 27, linewidth=1, edgecolor='red', facecolor='none')
    cuadrado2 = patches.Rectangle((1, 1), 25, 25, linewidth=1, edgecolor='red', facecolor='none')
    
    ax[fila,columna].add_patch(cuadrado1)
    ax[fila,columna].add_patch(cuadrado2)
    
    columna += 1 
 
# %%

# grficamos la distribución promedio en el eje x e y

def distribucion_promedio_eje_x(numero, titulo, ax):
    img = np.array(numero).reshape((28,28))
    val_img_prom = []
    
    for i in range(28):
        val_img_prom.append(img[:,i].mean())
    lista = list(range(1, 29))
    
    ax.bar(lista, val_img_prom)
    ax.set_title(titulo)


def distribucion_promedio_eje_y(numero, titulo, ax):
    img = np.array(numero).reshape((28,28))
    val_img_prom = []
    
    for i in range(28):
        val_img_prom.append(img[i].mean())
    lista = list(range(1, 29))
    
    ax.bar(lista, val_img_prom)
    ax.set_title(titulo)
    
# %%

# se saca el promedio de todos los pixeles de todos los números

promedio = mnist_manipulable.mean().to_frame()

# %%

# se grafica la imagen promedio
fig, ax = plt.subplots()

graficar_imagen(promedio, 'imagen promedio', ax)

# %%

# graficamos rectangulos agrupando pixeles a lo largo del eje x y del eje y

fig, ax = plt.subplots(1,2, figsize=(8,4))

graficar_imagen(promedio, 'imagen promedio \n con pixeles agrupados \n a lo largo del eje x', ax[0])

graficar_imagen(promedio, 'imagen promedio \n con pixeles agrupados \n a lo largo del eje y', ax[1])

for i in range(27):
    cuadrado = patches.Rectangle((i, 0), 1, 27, linewidth=1, edgecolor='yellow', facecolor='none')
    
    ax[0].add_patch(cuadrado)

for i in range(27):
    cuadrado = patches.Rectangle((0, i), 27, 1, linewidth=1, edgecolor='yellow', facecolor='none')
    
    ax[1].add_patch(cuadrado)

# %%

# representación del numero promedio, distribucion del mismo en el eje x e y

fig, ax = plt.subplots(1, 3, figsize=(15,4))

graficar_imagen(promedio, 'imagen promedio', ax[0])

distribucion_promedio_eje_x(promedio, 'distribución de valores \n de los píxeles en el eje x',  ax[1])
ax[1].set_ylim(-1, 1)

distribucion_promedio_eje_y(promedio, 'distribución de valores \n de los píxeles en el eje y', ax[2])
ax[2].set_ylim(-1, 1)


# %% b.
# %%

# comparacion entre las imagenes promedio del tres y el ocho y el tres y el uno


fig, ax = plt.subplots(2, 2, figsize=(6,6.5))

graficar_imagen(imagen_promedio(3), 'imagen promedio \n del número 3', ax[0,0])
graficar_imagen(imagen_promedio(1), 'imagen promedio \n del número 1', ax[0,1])
graficar_imagen(imagen_promedio(3), 'imagen promedio \n del número 3', ax[1,0])
graficar_imagen(imagen_promedio(8), 'imagen promedio \n del número 8', ax[1,1])

# %%

# representación de la resta de la imagen promedio del numero 3 y 1, distribucion de la misma en el eje x e y

comparacion_3_1 = imagen_promedio(3) - imagen_promedio(1)

fig, ax = plt.subplots(1, 3, figsize=(15,4))

graficar_imagen(comparacion_3_1, 'resta de las imagenes promedio \n del número 3 y 1', ax[0])

distribucion_promedio_eje_x(comparacion_3_1, 'distribución de valores \n de los píxeles en el eje x',  ax[1])
ax[1].set_ylim(-1, 1)

distribucion_promedio_eje_y(comparacion_3_1, 'distribución de valores \n de los píxeles en el eje y', ax[2])
ax[2].set_ylim(-1, 1)

# %%

# representación de la resta de la imagen promedio del numero 3 y 8, distribucion de la misma en el eje x e y

comparacion_3_8 = imagen_promedio(3) - imagen_promedio(8)

fig, ax = plt.subplots(1, 3, figsize=(15,4))

graficar_imagen(comparacion_3_8, 'resta de las imagenes promedio \n del número 3 y 8', ax[0])

distribucion_promedio_eje_x(comparacion_3_8, 'distribución de valores \n de los píxeles en el eje x',  ax[1])
ax[1].set_ylim(-1, 1)

distribucion_promedio_eje_y(comparacion_3_8, 'distribución de valores \n de los píxeles en el eje y', ax[2])
ax[2].set_ylim(-1, 1)

# %% c.
# %%

# matriz de ceros de muestra

mnist_ceros= mnist.query('labels == 0')
mnist_ceros_manipulable = mnist_ceros.drop('labels', axis=1)

fig, ax = plt.subplots(8, 8, figsize=(8,8))

for fila in range(8):
    for columna in range(8):
        graficar_imagen(mnist_ceros_manipulable.iloc[random.randint(0,len(mnist_ceros_manipulable)-1)], '', ax[fila,columna])

# %%

# imagen promedio del numero 0

fig, ax = plt.subplots()
graficar_imagen(imagen_promedio(0), 'imagen promedio del numero 0', ax)

# %%
#                                              CONSIGNA 2

# %% a
# %%

#filtramos el dataFrame para obtener imagenes solo del 0 y 1
archivo_filtered = mnist[mnist['labels'].isin([0, 1])]

#analizo cantidad de muestras por clases
class_counts = archivo_filtered['labels'].value_counts()

print("Numero de muestras para cada clase:")
print(class_counts)

#calculo la proporcion de cada clase
class_proportions = class_counts / len(archivo_filtered)

print("\nProporcion de cada clase")
print(class_proportions)

#determino si esta balanceado

balance = 0.1   #umbral de tolerancia
is_balanced = abs(class_proportions[0] - class_proportions[1]) < balance

if is_balanced:
    print("\nE1 conjunto de datos esta balanceado")
else:
    print("\nE1 conjunto de datos no esta balanceado")

# %% b
# %%

#separo las caracteristicas x y las etiquetas y
X = archivo_filtered.drop(columns = ['labels'])
Y = archivo_filtered['labels']

#divido los datos en conjuntos de entrenamiento y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)

#verifico el tamano de los conjuntos 
print(f"Numero de muestras en el conjunto de entrenamiento: {len(X_train)}")
print(f"Numero de muestras en el conjunto de testeo: {len(X_test)}")

# %% c
#%%

#armamos una funcion que nos da el numero de atributo a apartir de la coordenada del pixel
def cords (fila, columna):
    return str((fila - 1) * 28 + columna )

# %%

# creamos función para ajustar los conjuntos de knn

def knn_evaluation(atributos_elegidos):
    
    # creo un dicc para almacenar los resultados
    results = {}
    
    # evaluo cada conjunto de atributos
    for i, attributes in enumerate(atributos_elegidos): 
        
        # selecciono los atributos del conjunto actual
        X_train_subset = X_train[attributes]
        X_test_subset = X_test[attributes]
        
        # creo y entreno el modelo KNN
        knn = KNeighborsClassifier(n_neighbors=3)  # usamos K=3 como ejemplo
        knn.fit(X_train_subset, Y_train)
        
        # prediccion en el conjunto de prueba
        Y_pred = knn.predict(X_test_subset)
        
        # calculo la exactitud (accuracy)
        accuracy = accuracy_score(Y_test, Y_pred)
        results[f"Conjunto {i + 1}"] = accuracy
        
        print(f"Exactitud (Accuracy): {accuracy:.4f}")
    
    # resultados
    print("\nResultados de exactitud para cada conjunto de atributos:")
    for conjunto, accuracy in results.items():
        print(f"{conjunto}: {accuracy:.4f}")
    
# %%

# creamos función para marcar los pixeles en la imagen

def graficar_imagen_pixeles_marcados(numero, pixels_a_marcar, ax):
    # Crear una matriz de 28x28 llena de ceros
    img = np.array(numero).reshape((28, 28))
    
    # Expandir la matriz para añadir una nueva dimensión
    img_RGB = np.expand_dims(img, axis=-1)  # Ahora es de tamaño (28, 28, 1)
    
    # Repetir los valores a lo largo del nuevo eje para obtener (28, 28, 3)
    img_RGB = np.repeat(img_RGB, 3, axis=-1)  # Ahora es de tamaño (28, 28, 3)
    
    for (i, j) in pixels_a_marcar:
        img_RGB[i, j] = [255,0,0]
        
    ax.imshow(img_RGB, cmap='Reds')  # Usar alpha para transparencia
    ax.axis('off')

# %%

# defino conjuntos de atributos (píxeles)
atributos_elegidos_A = [
    [cords(14,13), cords(14,14), cords(14,15)], # Conjunto 1
    [cords(15,7), cords(15,14), cords(15,22)], # Conjunto 2
    [cords(22,12), cords(15,14), cords(6,16)] # Conjunto 3

]


fig, ax = plt.subplots(2,3)


graficar_imagen_pixeles_marcados(imagen_promedio(0), [(14,13), (14,14), (14,15)], ax[0,0]) # Conjunto 1
graficar_imagen_pixeles_marcados(imagen_promedio(0), [(15,7), (15,14), (15,22)], ax[0,1]) # Conjunto 2
graficar_imagen_pixeles_marcados(imagen_promedio(0),    [(22,12), (15,14), (6,16)], ax[0,2]) # Conjunto 3

graficar_imagen_pixeles_marcados(imagen_promedio(1), [(14,13), (14,14), (14,15)], ax[1,0]) # Conjunto 1
graficar_imagen_pixeles_marcados(imagen_promedio(1), [(15,7), (15,14), (15,22)], ax[1,1]) # Conjunto 2
graficar_imagen_pixeles_marcados(imagen_promedio(1),    [(22,12), (15,14), (6,16)], ax[1,2]) # Conjunto 3

ax[0,0].set_title('combinación 1') 
ax[0,1].set_title('combinación 2')
ax[0,2].set_title('combinación 3')

knn_evaluation(atributos_elegidos_A)

# %%


# defino conjuntos de atributos (píxeles)
atributos_elegidos_B = [
    [cords(15,7), cords(15,14), cords(15,22), cords(10,10)], # Conjunto 1
    [cords(15,7), cords(15,14), cords(15,22), cords(10,10), cords(22,12)], # Conjunto 1
    [cords(15,7), cords(15,14), cords(15,22), cords(10,10), cords(22,12), cords(6,16)] # Conjunto 1
]


fig, ax = plt.subplots(2,3)


graficar_imagen_pixeles_marcados(imagen_promedio(0), [(15,7), (15,14), (15,22), (10,10)], ax[0,0]) # Conjunto 1
graficar_imagen_pixeles_marcados(imagen_promedio(0), [(15,7), (15,14), (15,22), (10,10), (22,12)], ax[0,1]) # Conjunto 1
graficar_imagen_pixeles_marcados(imagen_promedio(0), [(15,7), (15,14), (15,22), (10,10), (22,12), (6,16)], ax[0,2]) # Conjunto 1

graficar_imagen_pixeles_marcados(imagen_promedio(1), [(15,7), (15,14), (15,22), (10,10)], ax[1,0]) # Conjunto 1
graficar_imagen_pixeles_marcados(imagen_promedio(1), [(15,7), (15,14), (15,22), (10,10), (22,12)], ax[1,1]) # Conjunto 1
graficar_imagen_pixeles_marcados(imagen_promedio(1), [(15,7), (15,14), (15,22), (10,10), (22,12), (6,16)], ax[1,2]) # Conjunto 1

ax[0,0].set_title('combinación 1') 
ax[0,1].set_title('combinación 2')
ax[0,2].set_title('combinación 3')

knn_evaluation(atributos_elegidos_A)

# %% d
# %%

# definimos los valores de k a probar para los 3 primeros conjuntos
k_values = [1, 3, 5, 7, 9]

# creamos un DataFrame vacío para almacenar los resultados
results_df = pd.DataFrame(columns=['Conjunto', 'k', 'Exactitud'])

# evaluamos cada combinación de atributos y k
for i, attributes in enumerate(atributos_elegidos_A):
    for k in k_values:
        print(f"\nEvaluando el conjunto de atributos {i + 1} con k={k}: {attributes}")
        
        # seleccionamos los atributos del conjunto actual
        X_train_subset = X_train[attributes]
        X_test_subset = X_test[attributes]
        
        # creamos y entrenamos el modelo KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_subset, Y_train)
        
        # predecimos en el conjunto de prueba
        Y_pred = knn.predict(X_test_subset)
        
        # calculamos la exactitud (accuracy)
        accuracy = accuracy_score(Y_test, Y_pred)
        
        # creamos un DataFrame temporal con los resultados actuales
        temp_df = pd.DataFrame({
            'Conjunto': [f"Conjunto {i + 1}"],
            'k': [k],
            'Exactitud': [accuracy]
        })
        
        # concatenamos el DataFrame temporal con el DataFrame principal
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
        
        print(f"Exactitud (Accuracy): {accuracy:.4f}")

# vemos los resultados
print("\nResultados de exactitud para cada combinación de atributos y k:")
print(results_df)

# %%


# Graficamos accuracy vs numero de vecinos (k)
# separamos con respecto a cada conjunto

for conjunto in results_df['Conjunto'].unique():
    subset = results_df[results_df['Conjunto'] == conjunto]
    plt.plot(subset['k'], subset['Exactitud'], marker='o', linestyle='-', label=conjunto)


plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Exactitud')
plt.legend(title="Conjunto de Atributos")
plt.grid()
plt.show()


# %%
#                                              CONSIGNA 3

# %%
#a
#%%

X = mnist.drop(columns=['labels'])  
y = mnist["labels"]  # etiquetas (dígitos entre 0 y 9)

# separamos el conjunto de datos en desarrollo (dev) y validación (held-out)

X_dev, X_heldout, y_dev, y_heldout = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# antes que nada, verificamos la distribución de clases para ver si estan balanceados
print("Distribución de clases:")
print(y_dev.value_counts()) 
#el dataset está relativamente balanceado, no hay grandes diferencias entre cada conjunto

#%%
#b
#%%

#realizamos el decision tree con algunas profundidades porque si usabamos del 1 al 10 nos tardaba mucho en correr
profundidades = [2,4,6,8,10]  # del 1 a 10

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

resultados = np.zeros((5, len(profundidades)))

for i, (train_index, test_index) in enumerate(kf.split(X_dev, y_dev)):
    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]

    for j, hmax in enumerate(profundidades):
        arbol = DecisionTreeClassifier(max_depth=hmax, random_state=42)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        resultados[i, j] = accuracy_score(kf_y_test, pred)
        
        
#%%
# calculamos el promedio de accuracy para cada profundidad
scores_promedio = resultados.mean(axis=0)

# graficamos accuracy vs max_depth
plt.figure(figsize=(8, 5))
plt.plot(profundidades, scores_promedio, marker='o', linestyle='-', color='b')
plt.xlabel('Profundidad del Árbol (max_depth)')
plt.ylabel('Accuracy Promedio')
plt.grid()
plt.show()

# obtenemos la mejor profundidad
mejor_hmax = profundidades[np.argmax(scores_promedio)]
print(f" Mejor profundidad seleccionada: {mejor_hmax}")


#%%
#c
#%%

#elegimos esta serie de hiper parametros para comparar modelos

param_grid = {
    'max_depth': range(1, 10),  # Profundidades de 1 a 10
    'min_samples_split': [2, 5],
    'min_samples_leaf': [2, 5],
    'criterion': ['gini']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs= -1
)
grid_search.fit(X_dev, y_dev)

# obtenemos el mejor modelo y sus hiperparámetros
mejor_modelo = grid_search.best_estimator_
mejor_params = grid_search.best_params_
mejor_score = grid_search.best_score_

print("\n Mejor modelo encontrado:")
print(mejor_modelo)
print("\n Hiperparámetros óptimos:")
print(mejor_params)
print(f"\n Accuracy en validación cruzada: {mejor_score:.4f}")

#%%
#d
#%%

#con los valores de held-out prececimos que digitos obtendremos y calculamos la performance
#la accuracy es un poco mejor que en el conjunto de dev

mejor_modelo.fit(X_dev, y_dev)
y_pred_heldout = mejor_modelo.predict(X_heldout)
score_heldout = accuracy_score(y_heldout, y_pred_heldout)
print(f"\n Precisión en el conjunto de validación (held-out): {score_heldout:.4f}")

# hacemos nuestra matriz de confusión
conf_matrix = confusion_matrix(y_heldout, y_pred_heldout)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

# por ultimo, tenemos nuestro reporte de clasificación, que nos muestra diferentes hiper parametros con respecto 
# a las diferentes profundidades utilizadas
reporte = classification_report(y_heldout, y_pred_heldout, target_names=[str(i) for i in range(10)])
print("\n Reporte de clasificación:\n")
print(reporte)


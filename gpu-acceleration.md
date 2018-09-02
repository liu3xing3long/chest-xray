# Aceleración de algoritmos de machine learning desde un enfoque arquitectónico

Se desarrollará un estudio detallado sobre el comportamiento de una red neuronal profunda de convolucion (CNN) sobre una arquitectura con GPU utilizando los datos obtenidos de Kaggle de https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. 

## Problema a tratar

Se ha seleccionado el dataset chest xray pneumonia, el cual esta compuesto por imagenes de rayos x de personas etiquetados segun tengan o no pneunomia. Se busca lograr la mayor precision en el mejor tiempo posible para la prediccion del label utilizando GPU para mejorar los tiempos de entrenamiento e inferencia.

## Software y Hardware

Se utilizó el lenguaje Python con la libreria Tensorflow para el diseño de la red neuronal.
Tambien se utilizaron pandas para cargar datos y procesarlos, numpy para operaciones sobre arreglos, skimage para el procesamiento de las imagenes y sklearn para su normalización.
Para correr la red neuronal se utilizo una CPU Intel Xeon E5-1620 v3 y una GPU Tesla K20, la cual fue usada con cuda 9 para acelerar el procesamiento de la red convolucional en tensorflow.

## Analisis de datos

Descargaremos los datos de https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia y los pondremos en la carpeta input.

```
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

Importaremos las librerias necesarias:
```
import os
from skimage import io, color, exposure
from skimage.transform import resize
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline 
```

Luego importaremos los datos:
```
test_normal_dir = "../input/test/NORMAL"
test_pneumonia_dir = "../input/test/PNEUMONIA"
train_normal_dir = "../input/train/NORMAL"
train_pneumonia_dir = "../input/train/PNEUMONIA"
full_url = np.vectorize(lambda url,prev_url: prev_url+"/"+url)
test_normal_data = pd.DataFrame(full_url(np.array(os.listdir(test_normal_dir)),test_normal_dir), columns=["image_dir"])
test_pneumonia_data = pd.DataFrame(full_url(np.array(os.listdir(test_pneumonia_dir)),test_pneumonia_dir), columns=["image_dir"])
train_normal_data = pd.DataFrame(full_url(np.array(os.listdir(train_normal_dir)),train_normal_dir), columns=["image_dir"])
train_pneumonia_data = pd.DataFrame(full_url(np.array(os.listdir(train_pneumonia_dir)),train_pneumonia_dir), columns=["image_dir"])
test_normal_data["class"] = "NORMAL"
test_pneumonia_data["class"] = "PNEUNOMIA"
train_normal_data["class"] = "NORMAL"
train_pneumonia_data["class"] = "PNEUNOMIA"
test_data = test_normal_data.append(test_pneumonia_data)
train_data = train_normal_data.append(train_pneumonia_data)
```

El tamaño de los datos de entrenamiento y test:
```
print("Training data size",train_data.shape)
print("Test data size",test_data.shape)```
```

```
Training data size (5216, 2)
Test data size (624, 2)
```

Repartimiento de clases en datos de entrenamiento:
```
# train clases sizes
pd.DataFrame(train_data['class'].value_counts())
```
  class   | ammount
----------|-------
PNEUNOMIA |	3875
NORMAL    |	1341

Repartimiento de clases en datos de test:
```
# test clases sizes
pd.DataFrame(test_data['class'].value_counts())
```
  class   | ammount
----------|-------
PNEUNOMIA |	390
NORMAL    |	234

En ambos casos se puede observar que existe un desbalance en las clases de las imagenes, ese problema sera resuelto más adelante por la utilizacion de diversas tecnicas, entre ellas la aplicacion de pesos segun la proporcion de elementos de cada clase a la predicción.

20 imagenes seleccionadas al azar

![some_images](https://github.com/okason97/chest-xray/blob/master/images/chest-images.png)

20 imagenes de personas con pneumonia

![some_images](https://github.com/okason97/chest-xray/blob/master/images/chest-images-PEUMONIA.png)

20 imagenes de personas sin pneumonia

![some_images](https://github.com/okason97/chest-xray/blob/master/images/chest-images-NORMAL.png)

## Preparación de datos

Dados los diferentes tamaños de las imagenes, estas serán modificadas para poseer todas el mismo tamaño (256x256).

Se reduciran dimensionalidad de las imagenes cambiando de canales RGB a un solo canal en escala de grises para reducir el tamaño del input con lo que las imagenes ocuparan menos espacio en memoria y podran ser procesadas más rapidamente.

Luego, se realizara ecualizacion de histograma para aumentar el contraste en las imagenes.

Finalmente, se normalizaran las imagenes, esto permite llegar más rapido a la convergencia. Se realizara MinMaxScaling, lo que escalara cada pixel de las imagenes al rango (-1,1) de la siguiente forma: 

```
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
```

20 imagenes normalizadas

![some_images](https://github.com/okason97/chest-xray/blob/master/images/chest-images-normalized.png)


## Definición del modelo

Se utilizara una CNN en tensorflow. Se probaron varias combinaciones de capas, 2,3 o 4 capas convolucionales y 1 o 2 capas fully connected. Luego de varios intentos con capas de 2 o 3 capas, se descartaron debido a que con estas cantidades de capas no se lograba extraer una cantidad suficiente de features, lo que generaba una baja accuracy (aproximadamente de 75% sobre los datos de test) y, en caso de alargar el entrenamiento, tendian a hacer overfitting, bajando la accuracy sobre lo datos de test aún más.
La selección de 2 capas fully connected tambien permitió un aumento en la accuracy sobre la seleccion de 1.
Finalmente se contará con una arquitectura con la siguiente forma, cada capa se explicará más adelante:

nombre de capa | descripcion
----- | -----
conv1 | capa convolucional con tamaño de ventana de 3x3, tamaño de input fmap de 1 , output fmap 16  y funcion de activacion leaky relu
conv2 | capa convolucional con tamaño de ventana de 3x3, tamaño de input fmap de 16 , output fmap 32  y funcion de activacion leaky relu
max_pool | max pooling con pasos de 2 x 2 quedando con imagenes de tamaño 128x128
conv3 | capa convolucional con tamaño de ventana de 3x3, tamaño de input fmap de 32 , output fmap 64  y funcion de activacion leaky relu
max_pool | max pooling con pasos de 2 x 2 quedando con imagenes de tamaño 64x64
conv4 | capa convolucional con tamaño de ventana de 3x3, tamaño de input fmap de 64 , output fmap 128 y funcion de activacion leaky relu
max_pool | max pooling con pasos de 2 x 2 quedando con imagenes de tamaño 32x32
dense1 | capa fully connected con 32x32x128 entradas y 512 salidas con funcion de activacion leaky relu
dense2 | capa fully connected con 512 entradas y 1024 salidas con funcion de activacion leaky relu
out | capa de salida usando cross entropy loss con softmax y prediccion pesada para contrarrestar el desbalance de clases de input

### Hiperparametros

Para la seleccion de los hiperparametros se corrieron diversas pruebas hasta encontrar aquellos que dieron los mejores resultados.
Se seleccionó un batch size de 64, el cual fue el que mayor velocidad daba a la red permitiendo el mayor paralelismo y no viendose limitado por falta de memoria disponible.
También se eligió setear learning rate con valor 0.01 que evitaba divergencia y permitio una convergencia aceptablemente rapida.
Por ultimo se inicio dropout con un valor de 0.75, de esta forma, en cada paso de la red, se deshabilitan un 25% de las neuronas de conv2, conv3, conv4, dense1 y dense2 lo cual permite evitar overfitting y le da mas robustez a la red.

### Capas convolucionales

Una red neuronal convolucional (CNN) es un tipo de red neuronal especialmente util en la clasificacion y reconocimiento de imagenes.
La principal funcionalidad de una CNN es la de extraer caracteristicas de las imagenes preservando la relacion espacial entre pixeles utilizando filtros, los cuales son matrices de pequeño tamaño, que se deslizaran por sobre las matrices de input. Cabe aclarar que es posible representar una imagen mediante una matriz donde cada pixel es un valor de esta.
A medida que el filtro se desplaza por la matriz input de a stride pixeles se multiplicaran los datos de la matriz input por los del filtro y luego se sumaran estos resultados para formar un elemento de la matriz de output, tambien llamada feature map.
Una vez formado el feature map se le sumara el bias para pasar el resultado a la proxima capa.
A medida que la red se entrena, este filtro ira cambiando y descubriendo nuevas features. 

![some_images](https://github.com/okason97/chest-xray/blob/master/images/convolutional.png)

Es posible la utilizacion de multiples canales de input mediante batchs lo que permite acelerar la velocidad de cada epoch de esta forma se puede usar el mismo filtro cacheado para multiples matrices de input simultaneamente. Tambien es posible aumentar la profundidad de la CNN aplicando multiples filtros simultaneamente a los mismos datos de entrada generando a su vez multiples feature maps. 

![some_images](https://github.com/okason97/chest-xray/blob/master/images/multiple-convolutional.png)

Es posible, gracias a lo mencionado anteriormente, representar la CNN como una multiplicacion de matrices usando Matrices de Toeplitz.

![some_images](https://github.com/okason97/chest-xray/blob/master/images/matrix-convolutional.png)

Las GPU son coprocesadores altamente segmentados con una gran cantidad de unidades funcionales y gran ancho de banda especialmente utiles para el trabajo paralelo. En el caso de la GPU utilizada para este proyecto se tiene un ancho de banda de 3.52 Tflops para operaciones de coma flotante de precision simple, 208 GBytes/s ancho de banda de memoria, 5 GB de memoria y 2496 nucleos CUDA. 
Todas estas caracteristicas vuelven a las GPU una herramienta sumamente útil para el procesamiento de redes neuronales, las cuales pueden tener sus operaciones representadas mediante multiplicación de matrices. La existencia de librerias de GPU como Nvidia CUBLAS, cuDNN, clBLAS, etc. permite utilizarlas para aumentar considerablemente la velocidad de procesamiento de la CNN.

![some_images](https://github.com/okason97/chest-xray/blob/master/images/gpu.jpg)

### Funciones de activacion

Para las funciones de activacion se decidio utilizar la funcion Leaky Relu, la cual tan como la funcion Relu presenta el mismo valor que el introducido en caso de ser positivo, pero a diferencia de Relu, Leaky Relu posee una pendiente para los valores menores a 0. Se decidio utilizar una pendiente con valor de alpha de 0.5, bastante mayor que la indicada en Empirical Evaluation of Rectified Activations in Convolution Network de 5.5 ya que luego de varias pruebas se descubrió que para este conjunto de datos, una pendiente mayor da mejores resultados.

![some_images](https://github.com/okason97/chest-xray/blob/master/images/leaky-relu.png)

### Pooling

Las capas pooling permiten reducir el tamaño de los datos que fluyen por la red neuronal. De esta forma es posible mejorar su performance y reducir overfitting al poseer menos informacion espacial.
La tecnica de pooling funciona mediante una ventana de tamaño k, la cual se desplazara por la matriz de datos. A medida que la ventana se desplaza de a pasos (stride), se realiza una funcion sobre todos los datos que la ventana abarque y el resultado sera un elemento de la matriz output, de esta forma el tamaño de la matriz sera reducido segun el valor de ventana que se tome y los pasos realizados.
Para pooling se probaron las funciones max y average, dando resultados similares, pero dado que max pooling otorgo resultados ligeramente superiores se opto por elegirla. 
Se eligió poner 3 capas de pooling, una entre la segunda y tercer capa convolucional, otra entre la tercer y cuarta capa convolucional, y otra más entre la cuarta capa convolucional y la primer capa fully connected. Esta elección se tomó para preservar en las primeras dos capas convolucionales la informacion espacial de las matrices de input. A medida que aumenta el tamaño de las capas, se realizara pooling para mantener la performance.

### Red fully connected

En una red fully connected cada neurona esta conectada con todas las de la siguiente capa y recibe todas las conexiones de la capa anterior. 
Es posible la aceleracion de esta capa mediante la utilizacion de GPU, ya que al utilizar batchs de datos, el procesamiento de la capa se transforma en una operacion de multiplicacion de matrices.

### Capa de salida

Al llegar a esta capa, los inputs se multiplican por el peso, se suman al bias y se suman entre si generando 2 outputs, uno para cada clase de persona (NORMAL y PNEUMONIA). Luego se aplicara a este resultado una multiplicacion por un peso segun la proporcion de elementos de cada clase en los datos, esto ayudara a contrarrestar el desbalance en las clases de los elementos y aumentara la accuracy sobre los datos de test. 
Al resultado anterior se le aplicara la funcion softmax para normalizarlo de forma tal que la suma de los outputs de softmax sean 1, se puede ver como la probabilidad de la pertenencia a cada clase. Utilizando este resultado, se aplicara cross entropy para conseguir un costo, este sera promediado con el resto de los costos calculados en el batch para concluir finalmente con la funcion de costo, la cual se buscara disminuir en cada iteracion de la red neuronal mediante backpropagation.

## Pipelining

La utilizacion de la herramienta Dataset provista por tensorflow permite la generacion de complejos pipelines para alimentar a la GPU con datos y evitar cuello de botella en la recoleccion de estos.
Para realizar la alimentacion de datos, se utiliza la CPU, la cual tendra la tarea de cargar las imagenes del disco, decodificarlas y procesar el tensor para poder ser procesado por la red, la cual sera ejecutada por la GPU.
Los datos seran mezclados por la CPU en cada epoch, tambien seran mapeados por esta en multiples hilos en paralelo para preparar las imagenes.
Adicionalmente, la CPU hara un prefetch de datos, es decir tomará datos de forma adelantada a que estos sean procesados, si la GPU esta procesando el batch n, la CPU, en ese momento, procesara el batch n+1.

![some_images](https://github.com/okason97/chest-xray/blob/master/images/pipelining.png)

El tiempo utilizando pipelining para el primer epoch fue de aproximadamente 56.7 segundos, en los epochs posteriores este tiempo bajo a aproximadamente 53.7 segundos debido al cacheo de las variables. En cambio, sin pipelining el tiempo se reduce a aproximadamente 59.4 segundos en el primer epoch y 56.1 segundos en los epoch posteriores. Por lo tanto la utilizacion de pipelining nos da un aumento en la velocidad de aproximadamente 3 segundos por epoch, un 1.6% de mejora.

## Entrenamiento

Para el entrenamiento se utilizo Adam optimizer, el cual presenta ventajas frente al optimizador de descenso de gradiente, como la inclusion de la tecnica del momentum, la cual busca evitar resultados optimos locales. La desventaja del optimizador Adam es que es mas lento que el descenso de gradiente, pero es mas sencillo de implementar y genera buenos resultados en problemas complejos.

### Resultado

Como resultado del entrenamiento se obtuvo una presicion en 26 epochs de 98.24% sobre los datos de entrenamiento y 85.94% sobre los datos de test.

## Uso de GPU

Utilizando la herramienta nvidia-smi, mediante el comando:

```
nvidia-smi -l 2
```
Es posible medir la utilizacion del gpu mientras corre la red neuronal.
Los resultados de esta medicion son un utilizamiento del 95.3% de la memoria de GPU, esto equivale a 4521MiB / 4743MiB con un utilizamiento de la GPU de entre 87% y 100%.

Tambien se midio el uso de la CPU mediante el comando

```
top
```

Dando un utilizamiento de la CPU mayor al 130% y una utilizacion de memoria de 1.8 GB aproximadamente.

Tiempo promedio por epoch con GPU 53.919642734527585 segundos

Tiempo promedio por epoch con CPU 399.6184551715851 segundos

## Conclusión

Poseer una GPU es un recurso muy importante a la hora de entrenar redes neuronales profundas, ya que la cantidad de operaciones en estas es sumamente grande y por lo tanto su tiempo de ejecucion sin hardware especializado se dispara, siendo en este caso el tiempo por epoch con GPU casi 8 veces menor que el tiempo por epoch con CPU. 
Es importante tambien la cooperacion de la CPU con la GPU proveyendo de un flujo constante de datos para asi evitar un efecto cuello de botella y aprovechar todo el potencial de la GPU.
Gracias a esto y a las diversas tecnicas utilizadas para evitar el desbalance de carga (aproximadamente 70/30) y el overfitting por la cantidad de datos limitada se pudo lograr una buena precision en un tiempo aceptable. 


## Fuentes
https://becominghuman.ai/image-data-pre-processing-for-neural-networks-498289068258

https://benanne.github.io/2015/03/17/plankton.html#prepro-augmentation

https://www.tensorflow.org/performance/datasets_performance

https://www.tensorflow.org/performance/performance_guide#optimizing_for_gpu

https://www.tensorflow.org/

http://www.holehouse.org/mlclass/10_Advice_for_applying_machine_learning.html

https://www.kth.se/social/files/588617ebf2765401cfcc478c/PHensmanDMasko_dkand15.pdf

https://www.datacamp.com/community/tutorials/tensorboard-tutorial

https://arxiv.org/pdf/1505.00853.pdf

https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/pooling_layer.html

https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

https://deepnotes.io/softmax-crossentropy

Material del curso Aceleración de Machine Learning desde un enfoque arquitectónico.

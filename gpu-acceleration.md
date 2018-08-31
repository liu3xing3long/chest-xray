# Aceleración de algoritmos de machine learning desde un enfoque arquitectónico

Se desarrollará un estudio detallado sobre el comportamiento de una red neuronal profunda de convolucion (CNN) sobre una arquitectura con GPU utilizando los datos obtenidos de Kaggle de https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia.

## Problema a tratar

Se ha seleccionado el dataset chest xray pneumonia y se intentara lograr la mayor precision y el mejor tiempo posible en la prediccion del label utilizando GPU.

### Analisis de datos

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
val_normal_dir = "../input/val/NORMAL"
val_pneumonia_dir = "../input/val/PNEUMONIA"
full_url = np.vectorize(lambda url,prev_url: prev_url+"/"+url)
test_normal_data = pd.DataFrame(full_url(np.array(os.listdir(test_normal_dir)),test_normal_dir), columns=["image_dir"])
test_pneumonia_data = pd.DataFrame(full_url(np.array(os.listdir(test_pneumonia_dir)),test_pneumonia_dir), columns=["image_dir"])
train_normal_data = pd.DataFrame(full_url(np.array(os.listdir(train_normal_dir)),train_normal_dir), columns=["image_dir"])
train_pneumonia_data = pd.DataFrame(full_url(np.array(os.listdir(train_pneumonia_dir)),train_pneumonia_dir), columns=["image_dir"])
val_normal_data = pd.DataFrame(full_url(np.array(os.listdir(val_normal_dir)),val_normal_dir), columns=["image_dir"])
val_pneumonia_data = pd.DataFrame(full_url(np.array(os.listdir(val_pneumonia_dir)),val_pneumonia_dir), columns=["image_dir"])
test_normal_data["class"] = "NORMAL"
test_pneumonia_data["class"] = "PNEUNOMIA"
train_normal_data["class"] = "NORMAL"
train_pneumonia_data["class"] = "PNEUNOMIA"
val_normal_data["class"] = "NORMAL"
val_pneumonia_data["class"] = "PNEUNOMIA"
test_data = test_normal_data.append(test_pneumonia_data)
train_data = train_normal_data.append(train_pneumonia_data)
val_data = val_normal_data.append(val_pneumonia_data)
```

El tamaño de los datos de entrenamiento, test y validacion:
```
print("Training data size",train_data.shape)
print("Test data size",test_data.shape)
print("Val data size",val_data.shape)
```
```
Training data size (5216, 2)
Test data size (624, 2)
Val data size (16, 2)
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

Repartimiento de clases en datos de validacion:
```
# val clases sizes
pd.DataFrame(val_data['class'].value_counts())
```
  class   | ammount
----------|-------
PNEUNOMIA |	8
NORMAL    |	8

20 imagenes seleccionadas al azar:
![some_images](https://github.com/okason97/google-landmark-machine-learning-acceleration/blob/master/plots/someimages.png)

### Preparación de datos

Dados los diferentes tamaños de las imagenes, estas deberan ser modificadas para poseer todas el mismo tamaño (256x256).

Se reduciran dimensionalidad de las imagenes cambiando de canales RGB a un solo canal en escala de grises.

Luego, se realizara ecualizacion de histograma para aumentar el contraste en las imagenes.

Finalmente, se normalizaran las imagenes, esto permite llegar más rapido a la convergencia. Se realizara MinMaxScaling, lo que escalara cada pixel de las imagenes al rango (-1,1) de la siguiente forma: 

```
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
```


### Definición del modelo

Se utilizara una CNN en tensorflow. Esta contará con una arquitectura con la siguiente forma

nombre de capa | descripcion
----- | -----
conv1 | capa convolucional con tamaño de ventana de 3x3 y 
relu |
max_pool |
conv2 |
relu |
max_pool |
dense1 |
out |



## Fuentes
https://becominghuman.ai/image-data-pre-processing-for-neural-networks-498289068258
https://benanne.github.io/2015/03/17/plankton.html#prepro-augmentation

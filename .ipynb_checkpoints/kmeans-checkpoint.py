import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
ATRIBUTOS = wine.data.features 
#y = wine.data.targets 

# Selección de atributos para clustering:
atributos_seleccionados = ["Alcohol", "Malicacid", "Ash", "Alcalinity_of_ash", "Magnesium"]

# Eliminación de registros que tangan valores NULL en algúno de los atributos seleccionados:
datos_limpios = ATRIBUTOS.dropna(subset=atributos_seleccionados)

# Copia del dataframe, tomando en cuenta sólo los atributos seleccionados:
datos = datos_limpios[atributos_seleccionados].copy()
#print(datos)

'''
Se normalizan los objetos usando la técnica min-max, donde los valores de todos los atributos
se convierten para estar dentro del mismo rango, que en este caso es [1, 9].

Esto es necesario ya que los rangos de valores varían mucho entre los distintos atributos.
Los valores de Malicacid y Ash rondan entre 2 y 5, pero los de Alcohol y Alcalinity_of_ash
rondan entre 10 y 20, mientras que Magnesium anda de 80 en adelante. Aquellos atributos
con un rango más grande de valores (Magnesium, en este caso) influirán mucho en el cálculo 
de la distancia, por lo que deben ser normalizados para que cada atributo contribuya en 
magnitudes similares.
'''
datos = ((datos - datos.min()) / (datos.max() - datos.min())) * 9 + 1

#print(datos)


'''Centroides'''

def get_centroides_aleatorios(datos, k: int = 5):
  centroides = []
  for _ in range(k):
    #Apply: itera cada columna del datafreame
    #Sample: selecciona un valor aleatorio en una columna
    #Float: convierte el valor a float, ya que x.sample regresaría un dataframe.
    #centroide = panda series
    centroide = datos.apply(lambda columna: float(columna.sample()))
    centroides.append(centroide)

  #combina todos los centroides, que son panda seres individuales, en un sólo panda dataframe.
  return pd.concat(centroides, axis=1)

centroides = get_centroides_aleatorios(datos, 5)

def get_clusters(datos, centroides):
  distancias = centroides.apply(lambda x: np.sqrt(((datos - x) ** 2).sum(axis=1)))

  # encontrar el índice del valor mínimo en cada registro
  # o sea, encontrar el centroide más cercano para cada registro.
  return distancias.idxmin(axis=1)


def get_centroides_nuevos(datos, clusters, k): 
  '''
  Promedio geométrico
  1. Agrupar datos por cluster
  2. Por cada cluster, calcular el promedio geométrico de sus atributos 
  '''
  return datos.groupby(clusters).apply(lambda x: np.exp(np.log(x).mean())).T

clusters = get_clusters(datos=datos, centroides=centroides)
print(get_centroides_nuevos(datos=datos, clusters=clusters, k = 5))

def graficar_clusters(datos, clusters, centroides, iteracion):
  pca = PCA(n_components= 2)
  datos_2d = pca.fit_transform(datos)
  centroides_2d = pca.transform(centroides.T)
  clear_output(wait=True)
  plt.title(f'Iteración {iteracion}')
  plt.scatter(x=datos_2d[:,0], y=datos_2d[:,1], c=clusters)
  plt.scatter(x=centroides_2d[:,0], y=centroides_2d[:,1])
  plt.show()
  
MAX_ITERACIONES = 100
K = 3

centroides = get_centroides_aleatorios(datos=datos, k=K)
centroides_anteriores = pd.DataFrame()
iteracion = 1

while iteracion < MAX_ITERACIONES and not centroides.equals(centroides_anteriores):
  centroides_anteriores = centroides
  clusters = get_clusters(datos=datos, centroides=centroides)
  centroides = get_centroides_nuevos(datos=datos, clusters=clusters, k=K)
  graficar_clusters(datos=datos, clusters=clusters, centroides=centroides, iteracion=iteracion)
  iteracion += 1
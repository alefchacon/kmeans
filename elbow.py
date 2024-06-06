import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ucimlrepo import fetch_ucirepo 
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold

import kmeans

WCSS_array = [] 
k_array = []
for k in range(1, 10):
  WCSS_array.append(kmeans.init(max_iteraciones=100, k=k))
  k_array.append(k)

  
plt.plot(k_array, WCSS_array, marker='o', linestyle='-', color='b')
plt.title("Gráfica Elbow")
plt.xlabel('K')
plt.ylabel('WCSS')
plt.show()

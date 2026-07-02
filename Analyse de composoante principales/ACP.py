import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import decomposition

def ACP(df) :
    pca = decomposition.PCA(n_components=9)
    pca.fit(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,5))

    ax1.plot(np.arange(1, 10), pca.explained_variance_ratio_, marker='o')

    ax1.set_xlabel("Nombre de composantes principales")
    ax1.set_ylabel("Proportion de variance expliquée")
    ax1.set_title("Proportion de variance expliquée par chaque composante principale")

    pcs = pca.components_

    ax2.scatter(pcs[0], pcs[1])
    for (x_coordinate, y_coordinate, feature_name) in zip(pcs[0], pcs[1], df.columns):
        ax2.text(x_coordinate, y_coordinate, feature_name)                          
        
    ax2.set_xlabel("Contribution à la PC1")
    ax2.set_ylabel("Contribution à la PC2")
    ax2.set_title("Contribution de chacun des paramètres aux différentes composantes principales")

    plt.show()

    
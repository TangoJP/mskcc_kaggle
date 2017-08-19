import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib  import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def decomposition3D(X, classes, y=None, decomposer=PCA(n_components=3)):
    if y is not None:
        X_decomp = decomposer.fit_transform(X,y)
    else:
        X_decomp = decomposer.fit_transform(X)

    decomp_table = pd.DataFrame(index=classes)
    decomp_table['x'] = X_decomp.T[0]
    decomp_table['y'] = X_decomp.T[1]
    decomp_table['z'] = X_decomp.T[2]

    return decomp_table

def decomposition3DPlot(decomp_table, classes):
    fig = plt.figure(1, figsize=(8, 8))
    plt.clf()
    ax = Axes3D(fig, elev=18, azim=55)
    ax.scatter(decomp_table.x, decomp_table.y, decomp_table.z,
               c=classes, marker='o', s=30, cmap = cm.jet)
    return

def decomposition2DPlot(decomp_table, classes):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,5))
    ax1.scatter(decomp_table.x, decomp_table.y, c=classes,
                marker = 'o', s=30, cmap = cm.jet, label='PCA1-PCA2')
    ax2.scatter(decomp_table.y, decomp_table.z, c=classes,
                marker = 'o', s=30, cmap = cm.jet, label='PCA2-PCA3')
    ax3.scatter(decomp_table.z, decomp_table.x, c=classes,
                marker = 'o', s=30, cmap = cm.jet, label='PCA3-PCA1')

    ax1.set_xlabel('Component-1')
    ax1.set_ylabel('Component-2')
    ax2.set_xlabel('Component-2')
    ax2.set_ylabel('Component-3')
    ax3.set_xlabel('Component-3')
    ax3.set_ylabel('Component-1')

    return

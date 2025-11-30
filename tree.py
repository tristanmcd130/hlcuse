import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# load the distances matrix
distances = pd.read_pickle("distances.pkl")

# distances is a full NxN matrix — hierarchical clustering needs
# a condensed distance matrix (upper triangle, flattened)
condensed = []

cols = distances.columns.tolist()

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        condensed.append(distances.iloc[i, j])

condensed = np.array(condensed)

# perform hierarchical clustering (average linkage)
Z = linkage(condensed, method="average")

# plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=cols, orientation='right')
plt.title("Indo-European Language Tree (Sentence Embedding Distances)")
plt.tight_layout()
plt.show()

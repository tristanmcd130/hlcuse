import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# load the distances matrix
distances = pd.read_pickle("distances.pkl")

# distances is a full NxN matrix - hierarchical clustering needs
# a condensed distance matrix (upper triangle, flattened)
condensed = []

cols = distances.columns.tolist()

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        condensed.append(distances.iloc[i, j])

condensed = np.array(condensed)

# perform hierarchical clustering (average linkage)
Z = linkage(condensed, method="average")

# dendrograms can"t be colored by individual branches, only by links
colors = {
    "en": "r",
    "da": "r",
    "de": "r",
    "el": "y",
    "lt": "m",
    "sv": "r",
    "sk": "g",
    "pl": "g",
    "es": "b",
    "nl": "r",
    "sl": "g",
    "pt": "b",
    "lv": "m",
    "cs": "g",
    "bg": "g",
    "ro": "b",
    "it": "b",
    "fr": "b"
}
link_colors = {}
for i, row in enumerate(Z):
    if row[0] < len(cols):
        link_colors[i + len(cols)] = colors[cols[int(row[0])]]
    elif row[1] < len(cols):
        link_colors[i + len(cols)] = colors[cols[int(row[1])]]
    elif link_colors[int(row[0])] == link_colors[int(row[1])]:
        link_colors[i + len(cols)] = link_colors[int(row[0])]
    else:
        link_colors[i + len(cols)] = "k"

# plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=cols, orientation="left", link_color_func=lambda x: link_colors[x])
plt.title("Indo-European Language Tree (Sentence Embedding Distances)")
plt.tight_layout()
plt.show()

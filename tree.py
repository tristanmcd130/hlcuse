import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
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

# Convert to tree structure for LCA queries
root, node_list = to_tree(Z, rd=True)


def get_edge_distance(leaf1, leaf2):
    """
    Get the number of edges between two leaf nodes in the tree.
    
    Parameters:
    - leaf1, leaf2: label names (e.g., 'en', 'de')
    
    Returns: number of edges between the two leaves
    """
    n = len(cols)
    idx1 = cols.index(leaf1)
    idx2 = cols.index(leaf2)
    
    # Build path from leaf to root
    def path_to_root(leaf_idx):
        path = [leaf_idx]
        current = leaf_idx
        for node in node_list[n:]:  # internal nodes only
            if node.left.id == current or node.right.id == current:
                path.append(node.id)
                current = node.id
        return path
    
    path1 = path_to_root(idx1)
    path2 = path_to_root(idx2)
    
    # Find LCA (first common node in both paths)
    set2 = set(path2)
    for i, node in enumerate(path1):
        if node in set2:
            return i + path2.index(node)
    return -1


# Compute all pairwise edge distances and save to pkl
edge_distances = pd.DataFrame(index=cols, columns=cols)
for i, lang1 in enumerate(cols):
    for j, lang2 in enumerate(cols):
        if i == j:
            edge_distances.loc[lang1, lang2] = 0
        else:
            edge_distances.loc[lang1, lang2] = get_edge_distance(lang1, lang2)

edge_distances.to_pickle("edge_distances.pkl")
print("Saved edge distances to edge_distances.pkl")
print("\nEdge distances matrix:")
print(edge_distances.to_string())

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

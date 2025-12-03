import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

distances = pd.read_pickle("distances.pkl")
embedding = MDS(dissimilarity="precomputed")
X = embedding.fit_transform(distances)
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
plt.scatter(X[:, 0], X[:, 1], s=200, c=list(map(lambda x: colors[x], distances.columns.tolist())))
for i, label in enumerate(distances.columns.tolist()):
    plt.text(X[i, 0], X[i, 1], label, ha="center", va="center", c="w")
plt.title("2D Language Vectors")
plt.show()

distances.replace(0, float("nan"), inplace=True)
order = [
	"en", "da", "de", "sv", "nl", # Germanic
	"el", # Greek
	"lt", "lv", # Baltic
	"sk", "pl", "sl", "cs", "bg", # Slavic
	"es", "pt", "ro", "it", "fr" # Romance
]
distances = distances.loc[:, order].reindex(order)
sns.heatmap(distances)
plt.title("Average Angular Distance between Sentence Embeddings")
plt.show()
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_pickle("sentences.pkl")
model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda:7") # MAKE SURE YOU ONLY USE 1 GPU
embeddings = model.encode(df.values.flatten(), normalize_embeddings=True, show_progress_bar=True).reshape((len(df), len(df.columns), 768)) # each column is a language, each row has the equivalent sentences in each language
distances = np.zeros((len(df.columns), len(df.columns)))
for i in range(len(df.columns)):
	for j in range(len(df.columns)):
		if i < j:
			print((i, j))
			for k in range(len(df)):
				distances[i][j] += np.arccos(np.clip(np.dot(embeddings[k][i], embeddings[k][j]), -1, 1))
			distances[i][j] /= len(df)
			distances[j][i] = distances[i][j]

distances = pd.DataFrame(distances)
distances.rename(columns=dict(enumerate(df.columns)), inplace=True)
distances.rename(index=dict(enumerate(df.columns)), inplace=True)
distances.to_pickle("distances.pkl")
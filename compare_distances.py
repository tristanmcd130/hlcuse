import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ISO codes to gold CSV labels (lowercase, with typos as they appear)
CODE_TO_GOLD = {
    "en": "english",
    "sv": "swedish",
    "da": "danish",
    "de": "german ",  # note trailing space in CSV
    "nl": "dutch",
    "ro": "romanian",
    "fr": "french",
    "it": "italian",
    "es": "spanish",
    "pt": "portguese",  # typo in CSV
    "lv": "latvian",
    "lt": "lithuanian",
    "pl": "polish",
    "sk": "slovak",
    "cs": "czech",
    "sl": "slovenian",
    "bg": "bulgarian",
    "el": "greek",
}

# Load matrices
edge_distances = pd.read_pickle("edge_distances.pkl")
gold_distances = pd.read_csv("gold_distances.csv", index_col=0)

# Fix inconsistent spelling in gold CSV (row="portguese", col="portuguese")
gold_distances.index = gold_distances.index.str.strip()
gold_distances.columns = gold_distances.columns.str.strip()
gold_distances = gold_distances.rename(index={"portguese": "portuguese"}, columns={"portguese": "portuguese"})

# Update mapping to use corrected labels
CODE_TO_GOLD_FIXED = {k: v.strip().replace("portguese", "portuguese") for k, v in CODE_TO_GOLD.items()}

# Rename edge_distances to match gold labels
edge_renamed = edge_distances.rename(index=CODE_TO_GOLD_FIXED, columns=CODE_TO_GOLD_FIXED)

# Get common labels in gold order
common_labels = [l for l in gold_distances.index if l in edge_renamed.index]

# Align matrices
edge_aligned = edge_renamed.loc[common_labels, common_labels].astype(float)
gold_aligned = gold_distances.loc[common_labels, common_labels].astype(float)

# Compute difference matrix
diff_matrix = edge_aligned - gold_aligned

# Print summary stats
print("=== Distance Matrix Comparison ===")
print(f"Mean absolute diff: {np.abs(diff_matrix.values).mean():.3f}")
print(f"Max absolute diff: {np.abs(diff_matrix.values).max():.3f}")
print(f"Total exact matches: {(diff_matrix.values == 0).sum()} / {diff_matrix.size}")

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Generated edge distances
sns.heatmap(edge_aligned, annot=True, fmt=".0f", cmap="Blues", ax=axes[0],
            square=True, cbar_kws={"shrink": 0.8})
axes[0].set_title("Generated Edge Distances")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# Plot 2: Gold distances
sns.heatmap(gold_aligned, annot=True, fmt=".0f", cmap="Blues", ax=axes[1],
            square=True, cbar_kws={"shrink": 0.8})
axes[1].set_title("Gold Distances")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

# Plot 3: Difference (confusion matrix style)
max_diff = max(abs(diff_matrix.values.min()), abs(diff_matrix.values.max()), 1)
sns.heatmap(diff_matrix, annot=True, fmt=".0f", cmap="RdBu_r", ax=axes[2],
            center=0, vmin=-max_diff, vmax=max_diff,
            square=True, cbar_kws={"shrink": 0.8})
axes[2].set_title("Difference (Generated - Gold)\nRed=Over, Blue=Under")
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig("distance_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved comparison plot to distance_comparison.png")

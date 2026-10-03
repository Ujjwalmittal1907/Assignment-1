"""
Assignment 1: Evaluate word embeddings using Embedding Projector concept.
Simple demo - only what is asked:
  1. Semantic similarity (cat-dog)
  2. Syntactic similarity (run-running)
  3. Analogies (king - man + woman = queen)
Uses cosine similarity + 2D PCA plot (like Embedding Projector)
+ generates files for projector.tensorflow.org
Run: python assignment1.py
"""
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Small demo word embeddings (8 dimensions) ----
# Dims: [animal, vehicle, run_verb, walk_verb, royal, masculine, feminine, past_tense]
embeddings = {
    "cat":      np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "dog":      np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "lion":     np.array([0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "tiger":    np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "car":      np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "bus":      np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "run":      np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "running":  np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "runs":     np.array([0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "walk":     np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "walking":  np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
    "king":     np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
    "queen":    np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
    "man":      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    "woman":    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    "prince":   np.array([0.0, 0.0, 0.0, 0.0, 0.8, 1.0, 0.0, 0.0]),
    "princess": np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 1.0, 0.0]),
    "boy":      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "girl":     np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
}

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def most_similar(word, topn=3):
    scores = [(w, cosine(embeddings[word], v)) for w, v in embeddings.items() if w != word]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topn]

def analogy(a, b, c, topn=1):
    # a - b + c  e.g. king - man + woman
    vec = embeddings[a] - embeddings[b] + embeddings[c]
    scores = [(w, cosine(vec, v)) for w, v in embeddings.items() if w not in (a, b, c)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topn]

print("=== 1. Semantic Similarity ===")
print(f"cos(cat, dog) = {cosine(embeddings['cat'], embeddings['dog']):.3f}")
print(f"cos(cat, car) = {cosine(embeddings['cat'], embeddings['car']):.3f}")
print("most similar to 'cat':", most_similar("cat"))
print("Conclusion: cat is closer to dog than to car -> semantic similarity captured.\n")

print("=== 2. Syntactic Similarity ===")
print(f"cos(run, running) = {cosine(embeddings['run'], embeddings['running']):.3f}")
print(f"cos(run, walk) = {cosine(embeddings['run'], embeddings['walk']):.3f}")
print("most similar to 'run':", most_similar("run"))
print("Conclusion: run is closer to running than to walk -> syntactic similarity captured.\n")

print("=== 3. Analogy ===")
result = analogy("king", "man", "woman")
print("king - man + woman =", result)
print("Conclusion: king - man + woman is closest to queen -> analogy captured.\n")

# ---- 2D plot like Embedding Projector (PCA with numpy) ----
words = list(embeddings.keys())
X = np.stack([embeddings[w] for w in words])
X_centered = X - X.mean(axis=0)
_, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
X_2d = X_centered @ Vt[:2].T

plt.figure()
for i, w in enumerate(words):
    plt.scatter(X_2d[i, 0], X_2d[i, 1])
    plt.text(X_2d[i, 0] + 0.02, X_2d[i, 1] + 0.02, w)
plt.title("Word Embeddings (PCA 2D) - like Embedding Projector")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("Assignment_1_Embedding_Projector/projector_plot.png", dpi=150, bbox_inches="tight")
print("Saved plot: Assignment_1_Embedding_Projector/projector_plot.png")

# ---- Files for projector.tensorflow.org ----
with open("Assignment_1_Embedding_Projector/vectors.tsv", "w") as f:
    for w in words:
        f.write("\t".join(map(str, embeddings[w])) + "\n")
with open("Assignment_1_Embedding_Projector/metadata.tsv", "w") as f:
    for w in words:
        f.write(w + "\n")
print("Saved vectors.tsv and metadata.tsv -> upload at https://projector.tensorflow.org")

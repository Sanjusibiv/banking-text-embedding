import numpy as np
from numpy.linalg import norm

# Load vocabulary
vocab = []
with open("vocab.txt") as f:
    next(f)
    for line in f:
        vocab.append(line.split(",")[0])

# Load embeddings
emb = np.loadtxt("embeddings_cbow.csv", delimiter=",")

print("Banking Word Embedding Evaluation")
print("Type 'q' to quit\n")

while True:
    query = input("Enter query word: ").strip().lower()

    if query == "q":
        print("Exiting evaluation.")
        break

    if query not in vocab:
        print(f"'{query}' not found in vocabulary.\n")
        continue

    qi = vocab.index(query)
    sims = []

    for i, w in enumerate(vocab):
        sim = np.dot(emb[qi], emb[i]) / (norm(emb[qi]) * norm(emb[i]) + 1e-9)
        sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)

    print(f"\nQuery: {query}")
    for w, s in sims[1:6]:  # skip the query word itself
        print(f"  {w} : {s:.4f}")
    print()

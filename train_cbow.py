import csv
import argparse
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1
    return v

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.05)
args = parser.parse_args()

# Load vocab
vocab = []
with open("vocab.txt") as f:
    next(f)
    for line in f:
        vocab.append(line.split(",")[0])
V = len(vocab)

# Load dataset
X, y = [], []
with open("cbow_dataset.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        context = np.mean([one_hot(int(i), V) for i in row[:-1]], axis=0)
        X.append(context)
        y.append(one_hot(int(row[-1]), V))

X = np.array(X)
y = np.array(y)

model = Sequential([
    Dense(10, input_shape=(V,), activation="linear"),
    Dense(V, activation="softmax")
])

model.compile(
    optimizer=SGD(learning_rate=args.lr),
    loss="categorical_crossentropy"
)

history = model.fit(X, y, epochs=args.epochs, verbose=1)

# Save loss
with open("loss_cbow.txt", "w") as f:
    for l in history.history["loss"]:
        f.write(f"{l}\n")

# Save embeddings
embeddings = model.layers[0].get_weights()[0]
np.savetxt("embeddings_cbow.csv", embeddings, delimiter=",")

print("CBOW training completed")

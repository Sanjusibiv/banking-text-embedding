import csv
import argparse
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.05)
args = parser.parse_args()

vocab = []
with open("vocab.txt") as f:
    next(f)
    for line in f:
        vocab.append(line.split(",")[0])
V = len(vocab)


def one_hot(i):
    v = np.zeros(V)
    v[i] = 1
    return v

X, y = [], []
with open("skipgram_dataset.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        X.append(one_hot(int(row[0])))
        y.append(one_hot(int(row[1])))

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

with open("loss_skipgram.txt", "w") as f:
    for l in history.history["loss"]:
        f.write(f"{l}\n")

embeddings = model.layers[0].get_weights()[0]
np.savetxt("embeddings_skipgram.csv", embeddings, delimiter=",")

print("Skip-gram training completed")
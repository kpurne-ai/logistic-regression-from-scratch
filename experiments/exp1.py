import numpy as np
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)
from model.LogisticRegression import LogisticRegression

from utils.metrics import accuracy
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])

model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(X)
print("Predictions:", predictions)
print("Accuracy:", accuracy(y, predictions))

probabilities = model.sigmoid(np.dot(X, model.get_weights()) + model.get_bias())

print()

print("Probabilities:")

print(probabilities)

model.plot_loss()
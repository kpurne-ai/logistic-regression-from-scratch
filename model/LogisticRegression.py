import numpy as np
import matplotlib.pyplot as plt
class LogisticRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            loss = - (1 / n_samples) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            self.loss_history.append(loss)
        return self
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        predicted_classes = [1 if p > 0.5 else 0 for p in predictions]
        return np.array(predicted_classes)

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_loss_history(self):
        return self.loss_history
    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title('Loss over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
"""
Very simple neural net with only one layer to learn the `2x + 1` pattern
"""
import numpy as np
import matplotlib.pyplot as plt


def mse(yhat: int, y: int):
    return 0.5 * ((yhat - y) ** 2)


class Model:
    def __init__(self, ds: np.ndarray, epochs: int, lr: float) -> None:
        self.ds = ds
        self.epochs = epochs
        self.lr = lr
        self.W = np.random.random((1, 1))
        self.b = np.random.random((1, 1))
        self.dw = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.E = np.zeros((self.epochs,))

    def train(self):
        for i in range(self.epochs):
            losses = np.zeros((ds.shape[0]))
            for j in range(ds.shape[0]):
                m = self.ds[j]
                x, y = m[0], m[1]
                yhat = self.predict(x)
                self.dw = x * (yhat - y)
                self.db = yhat - y
                self.W -= np.dot(self.lr, self.dw)
                self.b -= np.dot(self.lr, self.db)
                losses[j] += mse(yhat, y)
            self.E[i] += np.mean(losses)

    def predict(self, x: int) -> int:
        return (np.dot(self.W, x) + self.b)[0][0]

    def show_history(self):
        plt.plot(self.E)
        plt.title(f"LR: {self.lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


def get_ds(m: int):
    x = np.random.rand(m, 1)
    y = (x * 2) + 1
    return x, y, np.hstack((x, y))


if __name__ == "__main__":
    x, y, ds = get_ds(100)
    model = Model(ds=ds, epochs=20, lr=0.06)
    model.train()
    print(model.predict(-10))
    model.show_history()

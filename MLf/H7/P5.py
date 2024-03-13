import numpy as np
import pickle
from sklearn.linear_model import LinearRegression


def load_data(file_path):
    with open(file_path, "rb") as fh:
        loaded_data = pickle.load(fh)
    return loaded_data


def flin2d(x, w0, w1, w2):
    return np.array(w0 + w1 * x[:, 0] + w2 * x[:, 1], dtype=int)


def predictEffect(x):
    data = load_data("P5_data.pkl")
    X_train, Y_train = data["X"], data["Y"]

    model = LinearRegression()
    model.fit(X_train, Y_train)

    y_pred = model.predict(x)

    return y_pred.flatten()


if __name__ == "__main__":
    X2d = np.array([[1, 3], [2, 2], [4, 1]])
    Y = flin2d(X2d, 5, 4, 1)
    print(Y)

    xp = np.array([[0, 0], [10, 10], [5, 10]])
    yp = predictEffect(xp)
    print("yp =", yp)

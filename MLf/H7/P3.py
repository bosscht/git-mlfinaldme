import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# Load the data from 'P3_data.pkl'
def load_data():
    with open("P3_data.pkl", "rb") as fh:
        loaded_data = pickle.load(fh)

    xs = loaded_data["X"]
    ys = loaded_data["Y"]

    return xs, ys


def DPolyPredict(x):
    xs, ys = load_data()
    x_train, xtest, y_train, ytest = train_test_split(xs, ys, test_size=0.30)

    model = fit_quadratic_model(x_train, y_train)
    x = np.array(x).reshape(-1, 1)
    y_pred = model.predict(x)

    return y_pred


# make a quadratic model
def fit_quadratic_model(xt, yt):
    model = make_pipeline(PolynomialFeatures(degree=7), LinearRegression())
    model.fit(xt.reshape(-1, 1), yt)
    return model


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xs, ys = load_data()
    x_train, xtest, y_train, ytest = train_test_split(xs, ys, test_size=0.30)

    plt.plot(x_train, y_train, "k.")
    xp = np.linspace(0, 3, 200)
    yp = DPolyPredict(xp)
    plt.plot(xp, yp, "r--")
    plt.show()

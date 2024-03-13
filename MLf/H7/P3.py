import numpy as np
import pickle

with open("P3_data.pkl", "rb") as f:
    data = pickle.load(f)

x_train = data["X"]
y_train = data["Y"]
degree = 14

coefficients = np.polyfit(x_train, y_train, degree)


def DPolyPredict(x):
    y_pred = np.polyval(coefficients, x)
    return y_pred


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.plot(x_train, y_train, "k.")
    xp = np.linspace(0, 3, 200)
    yp = DPolyPredict(xp)
    plt.plot(xp, yp, "r--")
    plt.show()

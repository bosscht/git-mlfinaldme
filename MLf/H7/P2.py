import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def load_data(file_path):
    with open(file_path, "rb") as fh:
        loaded_data = pickle.load(fh)
    return loaded_data

def QPredict(x):
    data = load_data("P2_data.pkl")
    xs, ys = data['X'], data['Y']

    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)

    polynomial_features = PolynomialFeatures(degree=2)
    xs_poly = polynomial_features.fit_transform(xs)
    
    model = LinearRegression()
    model.fit(xs_poly, ys)

    x_input = x.reshape(-1, 1)
    x_input_poly = polynomial_features.transform(x_input)
    y_pred = model.predict(x_input_poly)

    return y_pred.flatten()

if __name__ == "__main__":
    xs = np.array([-10., -6.75, -3.5, -0.25, 3., 6.25, 9.5, 12.75, 16.])
    yp = QPredict(xs)
    print(yp)

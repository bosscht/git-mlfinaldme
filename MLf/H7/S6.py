import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor

if __name__ == '__main__':
    with open("S6.pkl", "rb") as fh:
        S6 = pickle.load(fh)

    print(sorted(list(S6.keys())))
    print(S6['x_train'].shape)
    print(S6['y_train'].shape)
    print(S6['x_val'].shape)
    print(S6['y_val'].shape)

    print(len(S6['weights']))
    print(len(S6['biases']))

    mod = MLPRegressor(hidden_layer_sizes=50)
    mod.coefs_ = S6["weights"]
    mod.intercepts_ = S6["biases"]
    mod.n_layers_ = 3
    mod.out_activation_ = 'identity'

    with open("P6_data.pkl", "rb") as fh:
        p6_data = pickle.load(fh)

    x_ = p6_data['X']
    y_ = p6_data['Y']

    yp = mod.predict(x_)
    mse = np.mean((yp - y_) ** 2)
    print('MSE =', mse)

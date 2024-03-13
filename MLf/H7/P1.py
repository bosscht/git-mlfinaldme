import numpy as np

def fquad(x, w):
    if isinstance(x, (int, float)):
        x = [x]
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    
    w = np.array(w)
    
    return w[0] + w[1]*x + w[2]*x**2


def mse(Yp, Y):
    Yp = np.array(Yp)
    Y = np.array(Y)
    
    return np.mean((Yp - Y)**2)

if __name__ == "__main__":
    xs = np.array([-4., -2., 0., 2., 4., 6., 8.])
    ys = fquad(xs, [5, 1, 1.5])
    print(ys)

    xt = np.array([-4., -2., 0., 2., 4., 6., 8.])
    yt = np.array([33., 11., 5., 15., 41., 83., 141.])
    yp = fquad(xt, [5, 1, 1.5])
    print(mse(yp, yt))

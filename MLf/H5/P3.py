import numpy as np


def normalize(v):
    n = np.linalg.norm(v)
    return v / n


if __name__ == "__main__":
    v = np.array([3, 4])
    u = normalize(v)
    print("u =", u)
    print("u.T @ v =", u.T @ v)
    print("size u =", np.linalg.norm(u))

    v = np.array([3, 4, 2])
    u = normalize(v)
    print("u =", u)
    print("u.T @ v =", u.T @ v)
    print("size u =", np.linalg.norm(u))

    v = np.array([3, 4, 2, 5])
    u = normalize(v)
    print("u =", u)
    print("u.T @ v =", u.T @ v)
    print("size u =", np.linalg.norm(u))

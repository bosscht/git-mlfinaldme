import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def eDiag(X):
    factors = np.load("P7_x.npy")
    labels = np.load("P7_y.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        factors, labels, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X)
    return predictions


if __name__ == "__main__":
    x = np.array([[5, 65, 4, 5, 3], [4, 72, 1, 3, 3], [2, 32, 2, 3, 3]])

    y = eDiag(x)
    print("Prediction:", y)

    Words = ["negative", "positive"]

    for i, yi in enumerate(y):
        print(f"Patient {i}'s result is {Words[yi]}.")

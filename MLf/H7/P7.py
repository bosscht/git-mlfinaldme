import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def eDiag(x):
    factors = np.load("P7_x.npy")
    labels = np.load("P7_y.npy")

    labels = np.ravel(labels)
    # Split the data into training and testing sets

    # Build a binary classification model
    model = RandomForestClassifier()
    model.fit(factors, labels)

    # Make predictions on the provided data
    predictions = model.predict(x)

    return predictions


# Test the eDiag function
if __name__ == "__main__":
    x = np.array([[5, 65, 4, 5, 3], [4, 72, 1, 3, 3], [2, 32, 2, 3, 3]])

    y = eDiag(x)
    print("Prediction:", y)
    Words = ["negative", "positive"]
    for i, yi in enumerate(y):
        print(f"Patient {i}'s result is {Words[yi]}.")

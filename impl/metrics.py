import random
from numpy.typing import NDArray


def train_test_split(
    X: NDArray,
    y: NDArray,
    training_samples: float,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    assert len(X) == len(y)

    examples = len(X)

    training = round(examples * training_samples)

    X_train, y_train = X[:training], y[:training]
    X_test, y_test = X[training:examples], y[training:examples]

    return (X_train, y_train, X_test, y_test)

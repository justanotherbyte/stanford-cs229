from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray
from IPython.display import clear_output


class GradientDescentAscent:
    def __init__(
        self,
        gradient_function: Callable[[NDArray], Union[NDArray, float]],
        *,
        learning_rate: float = 0.01,
        max_iter: int = 100_000,
        log_params: bool = False,
    ):
        self.gradient = gradient_function
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self._log_params = log_params

    @staticmethod
    def yield_batches(batch_size: int, *, feature_data: NDArray, label_data: NDArray):
        while True:
            for i in range(0, len(feature_data), batch_size):
                yield (
                    feature_data[i : i + batch_size],
                    label_data[i : i + batch_size],
                )

    def minimize(self, initial: NDArray) -> NDArray:
        converged = False
        iterations = 0

        params = initial

        while (iterations <= self.max_iter) and (converged is False):
            new = params - (self.learning_rate * self.gradient(params))

            if np.array_equal(params, new):
                converged = True

            params = new
            iterations += 1

            if self._log_params:
                print(params)

        if self._log_params:
            clear_output(wait=False)

        return params

    def maximize(self, initial: NDArray) -> NDArray:
        converged = False
        iterations = 0

        params = initial

        while (iterations <= self.max_iter) and (converged is False):
            new = params + (self.learning_rate * self.gradient(params))

            if np.array_equal(params, new):
                converged = True

            params = new
            iterations += 1

            if self._log_params:
                print(params)

        if self._log_params:
            clear_output(wait=False)

        return params

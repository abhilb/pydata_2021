from abc import ABC, abstractmethod
from pathlib import Path

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

import random
import numpy as np


class ClassificationModel(ABC):
    """
    Base class for the Classification Models
    """

    def __init__(self) -> None:
        super().__init__()

        self.train_samples = 5000
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

        random_state = check_random_state(0)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        X = X.reshape((X.shape[0], -1))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=self.train_samples, test_size=10000
        )

        self.model = None

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def get_score(self) -> float:
        return self.model.score(self.X_test, self.y_test)

    @abstractmethod
    def to_onnx(self, model_path: Path):
        pass

    @abstractmethod
    def to_treelite(self, lib_path: Path):
        pass

    def get_random_image(self):
        idx = random.randint(0, self.y_test.shape[0])
        img = np.reshape(self.X_test[idx], (28, 28))
        img = img * 255
        return img.astype(np.uint8)

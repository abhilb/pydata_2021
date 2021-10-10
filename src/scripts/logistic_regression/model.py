"""
Script to create a Logistic Regression Model for MNIST data
"""

from classification_model import ClassificationModel
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class LogisticRegresssionModel(ClassificationModel):
    def create_model(self):

        # Turn up tolerance for faster convergence
        self.model = LogisticRegression(
            C=50.0 / self.train_samples, penalty="l1", solver="saga", tol=0.1
        )
        self.model.fit(self.X_train, self.y_train)
        return super().create_model()

    def to_onnx(self):
        initial_type = [("float_input", FloatTensorType([None, 4]))]
        onx = convert_sklearn(self.model, initial_types=initial_type)

        with open("log_reg_model.onnx", "wb") as f:
            f.write(onx.SerializeToString())

        return super().to_onnx()

from classification_model import ClassificationModel
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class SVMClassifierModel(ClassificationModel):
    def create_model(self):
        self.model = svm.SVC(decision_function_shape="ovo")
        scaler = MinMaxScaler()
        X = scaler.fit_transform(self.X_train)
        self.model.fit(X, self.y_train)
        return super().create_model()

    def get_score(self) -> float:
        return super().get_score()

    def to_onnx(self, model_path: Path):
        initial_type = [("float_input", FloatTensorType([None, 784]))]
        onx = convert_sklearn(self.model, initial_types=initial_type)

        with open(str(model_path), "wb") as f:
            f.write(onx.SerializeToString())

        return super().to_onnx(model_path)

    def to_treelite(self, lib_path: Path):
        return super().to_treelite(lib_path)

    def to_pmml(self, file_path: Path):
        return super().to_pmml(file_path)

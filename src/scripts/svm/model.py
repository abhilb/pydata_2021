from classification_model import ClassificationModel
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import make_pipeline


class SVMClassifierModel(ClassificationModel):
    def create_model(self):
        self.model = make_pipeline(
            StandardScaler(), svm.SVC(decision_function_shape="ovo")
        )
        self.model.fit(self.X_train, self.y_train)
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

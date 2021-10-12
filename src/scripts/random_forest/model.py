from classification_model import ClassificationModel
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class RandomForestModel(ClassificationModel):
    def create_model(self):
        self.model = RandomForestClassifier(
            n_estimators=20, max_depth=None, min_samples_split=2, random_state=0
        )
        self.model.fit(self.X_train, self.y_train)
        return super().create_model()

    def get_score(self) -> float:
        return super().get_score()

    def to_onnx(self, model_path: Path):
        initial_type = [("float_input", FloatTensorType([None, 4]))]
        onx = convert_sklearn(self.model, initial_types=initial_type)

        with open(str(model_path), "wb") as f:
            f.write(onx.SerializeToString())

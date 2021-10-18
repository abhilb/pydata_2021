from classification_model import ClassificationModel
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import make_pipeline

import treelite.sklearn


class RandomForestModel(ClassificationModel):
    def create_model(self):
        self.model = RandomForestClassifier(
            n_estimators=20, max_depth=None, min_samples_split=2, random_state=0
        )
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

    def to_treelite(self, lib_path: Path):
        treelite_model = treelite.sklearn.import_model_with_model_builder(self.model)
        treelite_model.export_lib(
            toolchain="gcc",
            libpath=str(lib_path),
            params={"parallel_comp": 8},
            verbose=True,
        )

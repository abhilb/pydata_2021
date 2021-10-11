from classification_model import ClassificationModel
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(ClassificationModel):
    def create_model(self):
        self.model = RandomForestClassifier(
            n_estimators=10, max_depth=None, min_samples_split=2, random_state=0
        )
        self.model.fit(self.X_train, self.y_train)
        return super().create_model()

    def get_score(self) -> float:
        return super().get_score()

    def to_onnx(self, model_path: Path):
        pass

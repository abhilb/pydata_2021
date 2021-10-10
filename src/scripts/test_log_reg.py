from logistic_regression.model import LogisticRegresssionModel
from pathlib import Path


def test_to_onnx():
    model = LogisticRegresssionModel()
    model_path = Path.cwd() / "log_reg_model.onnx"    
    model.create_model()
    model.to_onnx(str(model_path))
    assert model_path.exists()

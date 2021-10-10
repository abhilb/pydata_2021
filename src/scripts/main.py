from logistic_regression.model import LogisticRegresssionModel
from pathlib import Path


def main():
    log_reg_model = LogisticRegresssionModel()
    log_reg_model.create_model()
    log_reg_model.to_onnx(Path.cwd() / "log_reg_model.onnx")


if __name__ == "__main__":
    main()

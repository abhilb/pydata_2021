from os import get_exec_path
from logistic_regression.model import LogisticRegresssionModel
from random_forest.model import RandomForestModel
from svm.model import SVMClassifierModel
from pathlib import Path
import logging
import os

logging.basicConfig()
logger = logging.getLogger("DemoApp")
logger.setLevel(logging.INFO)


def main():
    logger.info("Logistic Regression Model")
    log_reg_model = LogisticRegresssionModel()
    log_reg_model.create_model()
    logger.info(f"Logistic Regression Score: {log_reg_model.get_score()}")
    log_reg_model.to_onnx(Path.cwd() / "onnx_models" / "log_reg_model.onnx")

    logger.info("Random Forest Model")
    random_forest_model = RandomForestModel()
    random_forest_model.create_model()
    logger.info(f"Random Forest Model Score: {random_forest_model.get_score()}")
    random_forest_model.to_onnx(Path.cwd() / "onnx_models" / "random_forest_model.onnx")

    lib_ext = ".dll" if os.name == "nt" else ".so"
    random_forest_model.to_treelite(
        lib_path=Path.cwd() / f"random_forest_model{lib_ext}"
    )

    logger.info("SVM Classifier Model")
    svm_model = SVMClassifierModel()
    svm_model.create_model()
    logger.info(f"SVM Model score: {svm_model.get_score()}")
    svm_model.to_onnx(Path.cwd() / "onnx_models" / "svm_model.onnx")


if __name__ == "__main__":
    main()

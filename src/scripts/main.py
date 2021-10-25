from os import get_exec_path
from logistic_regression.model import LogisticRegresssionModel
from random_forest.model import RandomForestModel, RandomForestPmmlModel
from svm.model import SVMClassifierModel
from pathlib import Path
import logging
import os
import argparse

logging.basicConfig()
logger = logging.getLogger("DemoApp")
logger.setLevel(logging.INFO)


def main(algo):
    if algo == "all" or algo == "logistic_regression":
        logger.info("Logistic Regression Model")
        log_reg_model = LogisticRegresssionModel()
        log_reg_model.create_model()
        logger.info(f"Logistic Regression Score: {log_reg_model.get_score()}")
        log_reg_model.to_onnx(
            Path.cwd().parent / "models" / "onnx_models" / "log_reg_model.onnx"
        )

    if algo == "all" or algo == "random_forest":
        logger.info("Random Forest Model")
        random_forest_model = RandomForestModel()
        random_forest_model.create_model()
        logger.info(f"Random Forest Model Score: {random_forest_model.get_score()}")
        random_forest_model.to_onnx(
            Path.cwd().parent / "models" / "onnx_models" / "random_forest_model.onnx"
        )

        lib_ext = ".dll" if os.name == "nt" else ".so"
        random_forest_model.to_treelite(
            lib_path=Path.cwd().parent
            / "models"
            / "treelite_models"
            / f"random_forest_model{lib_ext}"
        )

        pmml_model = RandomForestPmmlModel()
        pmml_model.create_model()
        pmml_model.to_pmml(
            Path.cwd().parent / "models" / "pmml_models" / "random_forest_model.pmml"
        )

    if algo == "all" or algo == "svm":
        logger.info("SVM Classifier Model")
        svm_model = SVMClassifierModel()
        svm_model.create_model()
        logger.info(f"SVM Model score: {svm_model.get_score()}")
        svm_model.to_onnx(
            Path.cwd().parent / "models" / "onnx_models" / "svm_model.onnx"
        )
        print(svm_model.model.get_params())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=["all", "svm", "random_forest", "logistic_regression"],
        default="all",
    )
    args = vars(parser.parse_args())
    main(args["algo"])

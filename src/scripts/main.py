from logistic_regression.model import LogisticRegresssionModel
from random_forest.model import RandomForestModel
from pathlib import Path
import logging

logging.basicConfig()
logger = logging.getLogger("DemoApp")
logger.setLevel(logging.INFO)


def main():
    logger.info("Logistic Regression Model")
    log_reg_model = LogisticRegresssionModel()
    log_reg_model.create_model()
    logger.info(f"Logistic Regression Score: {log_reg_model.get_score()}")
    log_reg_model.to_onnx(Path.cwd() / "log_reg_model.onnx")

    logger.info("Random Forest Model")
    random_forest_model = RandomForestModel()
    random_forest_model.create_model()
    logger.info(f"Random Forest Model Score: {random_forest_model.get_score()}")
    random_forest_model.to_onnx(Path.cwd() / "random_forest_model.onnx")


if __name__ == "__main__":
    main()

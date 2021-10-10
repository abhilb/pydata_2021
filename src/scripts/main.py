from logistic_regression.model import LogisticRegresssionModel


def main():
    log_reg_model = LogisticRegresssionModel()
    log_reg_model.create_model()
    log_reg_model.to_onnx()


if __name__ == "__main__":
    main()

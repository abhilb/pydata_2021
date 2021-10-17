from pathlib import Path
import onnxruntime as ort
import imageio
import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler


@pytest.fixture
def input_image():
    image_fname = Path(__file__).parent / "test.bmp"
    print(image_fname)
    return imageio.imread(image_fname)


def test_random_forest_pred(capsys, input_image):
    model_path = Path(__file__).parent / "onnx_models" / "random_forest_model.onnx"
    print(model_path)

    input_image = input_image.astype(np.float32)
    input_image = np.ravel(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    print(input_image.shape)

    sess = ort.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    probs_name = sess.get_outputs()[1].name
    preds, probs = sess.run([label_name, probs_name], {input_name: input_image})
    print(preds)
    print(probs)
    assert False

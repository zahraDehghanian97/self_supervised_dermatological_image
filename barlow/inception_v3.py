from utils.model_utils import load_model
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import experimental
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization


def load_inception(model_path):
    return load_model(model_path + 'inception')


def get_network():
    model_path = ''
    inputs = Input(shape=(299, 299, 3))
    x = experimental.preprocessing.Rescaling(scale=1.0 / 255.0)(inputs)

    inception = load_inception(model_path)
    x = inception(x)
    out = BatchNormalization()(x)
    return Model(inputs, out)


if __name__ == '__main__':
    model = get_network()

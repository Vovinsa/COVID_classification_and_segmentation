from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model

def create_model():
    resnet = ResNet152V2(include_top=False, weights="imagenet", input_shape=(256, 256, 3))

    x = Flatten()(resnet.output)
    x = Dense(512, activation="relu")(x)
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=resnet.input, outputs=predictions)

    return model

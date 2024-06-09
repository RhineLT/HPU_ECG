import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0

def EFFNet():
    base_model = EfficientNetV2B0(include_top=False, input_shape=(None, None, 3), weights='imagenet')

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model

# Example usage
model = AFNet()
model.summary()

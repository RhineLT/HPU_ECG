import tensorflow as tf
from tensorflow.keras import layers, models


def AFNet():
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Flatten(),  # 将卷积层输出扁平化处理，以便输入到全连接层
        layers.Dropout(0.5),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model

def MobileNetModel():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(1, 1250, 3), include_top=False)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model

def AFNet_Com():
    model = models.Sequential([
        layers.Conv2D(filters=8, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
#         layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=16, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
#         layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=32, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
#         layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=32, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
#         layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=32, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
#         layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(20, activation='relu'),
        layers.Dense(2)
    ])

    return model


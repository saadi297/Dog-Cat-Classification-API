import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class model:
    def __init__(self, input_shape, num_classes, is_data_aug=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_data_aug = is_data_aug

    def create_model(self):
        inputs = keras.Input(shape=self.input_shape)

        # Image augmentation block
        if self.is_data_aug:
            data_augmentation = keras.Sequential(
                [
                    layers.experimental.preprocessing.RandomFlip("horizontal"),
                    layers.experimental.preprocessing.RandomRotation(0.1),
                ]
            )
            x = data_augmentation(inputs)
        else:
            x = inputs

        # Entry block
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if self.num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = self.num_classes

        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units, activation=activation)(x)
        return keras.Model(inputs, outputs)



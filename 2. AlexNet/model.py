from tensorflow.keras import layers, models, Model, Sequential

"""
用两种方式构建模型
"""

def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32") # output (None, 224, 224, 3)
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)                     # output (None, 227, 227, 3)
    x = layers.Conv2D(96, kernel_size=11, strides=4, activation='relu')(x)      # output (None, 55, 55, 96)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                             # output (None, 27, 27, 96)
    x = layers.Conv2D(256, kernel_size=5, padding='same', activation='relu')(x) # output (None, 27, 27, 256)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                             # output (None, 13, 13, 256)
    x = layers.Conv2D(384, kernel_size=3, padding='same', activation='relu')(x) # output (None, 13, 13, 384)
    x = layers.Conv2D(384, kernel_size=3, padding='same', activation='relu')(x) # output (None, 13, 13, 384)
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x) # output (None, 13, 13, 256)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                             # output (None, 6, 6, 256)

    x = layers.Flatten()(x)                                                     # output (None, 6*6*256)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(9012, activation='relu')(x)                                # output (None, 9012)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(4096, activation='relu')(x)                                # output (None, 4096)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(4096, activation='relu')(x)                                # output (None, 4096)
    x = layers.Dense(num_classes)(x)                                            # output (None, num_classes))
    predict = layers.Softmax()(x)

    model = models.Model(inputs=input_image, outputs=predict)
    return model


class AlexNet_v2(Model):
    def __init__(self, num_classes=1000):
        super(AlexNet_v2, self).__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(((1, 2), (1, 2))),
            layers.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(256, kernel_size=5, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2)
        ])

        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dropout(0.2),
            layers.Dense(9012, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(4096, activation='relu'),
            layers.Dense(num_classes),
            layers.Softmax()
        ])

    def call(self, inputs, **kwargs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
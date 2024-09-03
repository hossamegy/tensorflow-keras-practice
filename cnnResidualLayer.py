from tensorflow.keras.layers import Layer, Conv2D

class CNNResidualLayer(Layer):
    def __init__(self, layers, filters, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Conv2D(filters, (3, 3), activation='relu', padding='same') for _ in range(layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x
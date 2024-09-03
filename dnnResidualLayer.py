from tensorflow.keras.layers import Layer, Dense

class DNNResidualLayer(Layer):
    def __init__(self, layers, units, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Dense(units, activation='relu') for _ in range(layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x
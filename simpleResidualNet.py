from tensorflow.keras.layers import Layer, Dense, Conv2D, Input
from tensorflow.keras import Model

class CNNResidualLayer(Layer):
    def __init__(self, layers, filters, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Conv2D(filters, (3, 3), activation='relu', padding='same') for _ in range(layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x
    
class DNNResidualLayer(Layer):
    def __init__(self, layers, units, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Dense(units, activation='relu') for _ in range(layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x
    
class SimpleResidualNet(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_1 = Dense(25, activation='relu')
        self.block_1 = CNNResidualLayer(2, 48)
        self.block_2 = DNNResidualLayer(2, 64)
        self.out = Dense(1)

    def call(self, inputs):
        x = self.hidden_1(inputs)
        x = self.block_1(x)
        for _ in range(0, 2):
           x = self.block_2(x)
        return self.out(x)


# Create the final model
final_model = SimpleResidualNet()

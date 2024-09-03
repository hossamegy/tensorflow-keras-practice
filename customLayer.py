from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow import  Variable, random_normal_initializer, zeros_initializer, matmul

class SimpleCustomLayer(Layer):
  def __init__(self, units=32, activation=None):
    super(SimpleCustomLayer, self).__init__()
    self.units = units
    self.activation = activations.get(activation)

  def build(self, input_shape):
    w_init = random_normal_initializer()
    self.w = Variable(name='kernal',
                         initial_value=w_init(shape=(input_shape[-1], self.units), 
                                              dtype='float32'),
                         trainable=True)
    b_init = zeros_initializer()
    self.b = Variable(name='bias',
                         initial_value=b_init(shape=(self.unit,), 
                                              dtype='float32'),
                         trainable=True)
    
    def call(self, inputs):
      return self.activation (matmul(inputs, self.w) + self.b)
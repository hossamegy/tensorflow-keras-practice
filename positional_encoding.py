import tensorflow as tf
from tensorflow.keras.layers import Layer

class PositionalEncoding(Layer):
    def __init__(self, max_len, dims_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.dims_model = dims_model
        self.pos_encoding = None

    def build(self, input_shape):
        positions = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]
        dims = tf.range(self.dims_model, dtype=tf.float32)

        denominator_even = tf.pow(10000.0, 2 * (dims[::2]) / self.dims_model)
        denominator_odd = tf.pow(10000.0, (2 * (dims[1::2]) + 1) / self.dims_model)

        even_pos = tf.sin(positions / denominator_even)
        odd_pos = tf.cos(positions / denominator_odd)

        self.pos_encoding = tf.concat([even_pos, odd_pos], axis=-1)
        self.pos_encoding = self.pos_encoding[tf.newaxis, :, :]

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

max_len = 50
dims_model = 64
pe_layer = PositionalEncoding(max_len, dims_model)
inputs = tf.random.uniform((1, max_len, dims_model))
outputs = pe_layer(inputs)
print(outputs.shape)
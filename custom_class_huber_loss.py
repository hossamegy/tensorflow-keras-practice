import tensorflow as tf
from tensorflow.keras.losses import Loss

class MyHuberLoss(Loss):
    def __init__(self, threshold=1.0, name='my_huber_loss'):
        super().__init__(name=name)
        self.threshold = threshold

    def call(self, y_target, y_predicted):
        error = y_target - y_predicted
        condition = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) * 0.5
        big_error_loss = self.threshold * (tf.abs(error) - (0.5 * self.threshold))
        return tf.where(condition, small_error_loss, big_error_loss)

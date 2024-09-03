import tensorflow as tf

def huber_loss(threshold):
    def huber_loss_func(y_target, y_predicted):
        error = y_target - y_predicted
        condition = tf.abs(error) <= threshold
        small_error_loss = tf.square(error) * 0.5
        big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
        return tf.where(condition, small_error_loss, big_error_loss)
    return huber_loss_func

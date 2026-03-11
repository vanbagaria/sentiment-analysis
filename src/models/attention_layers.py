import tensorflow as tf
from tensorflow.keras.layers import Layer

class SimpleAttention(Layer):

    def __init__(self):
        super(SimpleAttention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):

        score = tf.tanh(tf.matmul(inputs, self.W))
        weights = tf.nn.softmax(score, axis=1)

        context = weights * inputs
        context = tf.reduce_sum(context, axis=1)

        return context

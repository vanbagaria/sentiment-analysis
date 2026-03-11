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

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        self.W1 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, values):
        score = self.V(tf.nn.tanh(self.W1(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

class LuongAttention(Layer):
    def __init__(self):
        super(LuongAttention, self).__init__()

    def call(self, values):
        score = tf.matmul(values, values, transpose_b=True)
        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, values)
        context_vector = tf.reduce_mean(context_vector, axis=1)
        return context_vector


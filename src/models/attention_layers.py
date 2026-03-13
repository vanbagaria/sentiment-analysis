import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.saving import register_keras_serializable

@register_keras_serializable()
class SimpleAttention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def get_config(self):
        return super().get_config()

from keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow.keras.layers import Layer


@register_keras_serializable()
class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W1 = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, values):
        score = self.V(tf.nn.tanh(self.W1(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config

@register_keras_serializable()
class LuongAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, values):
        score = tf.matmul(values, values, transpose_b=True)
        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, values)
        context_vector = tf.reduce_mean(context_vector, axis=1)
        return context_vector

    def get_config(self):
        return super().get_config()


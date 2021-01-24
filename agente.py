import tensorflow as tf
import numpy as np
import lunar_lander.py
from lunar_lander.py import LunarLander

env = LunarLander()
nactions = env.action_space.n
gamma = 0.99

class MyModel(tf.keras.Model):
    def __init__(self, state, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=(state.shape())
        )
        self.hidden_layer = tf.keras.layer.Dense(
            hidden_units, activation='relu', kernel_initializer='RandomNormal')
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

# Output is three values, Q for all possible actions
Qs = MyModel(state, 64, nactions)

# Optimizer adjusts weights to minimize loss, with the speed of learning_rate
optimizer = tf.optimizers.Adam(1e-3)

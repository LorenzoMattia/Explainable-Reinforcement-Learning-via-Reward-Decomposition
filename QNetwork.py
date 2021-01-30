import tensorflow as tf

class QNetwork(tf.keras.Model):
    def __init__(self, hidden_units, num_actions):
        super(QNetwork, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            hidden_units, activation='relu', kernel_initializer='RandomNormal')
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.hidden_layer(inputs)
        output = self.output_layer(z)
        return output
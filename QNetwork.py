import tensorflow as tf

class QNetwork(tf.keras.Model):
    def __init__(self, hidden_units, num_actions):
        super(QNetwork, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            hidden_units, activation='relu')
        self.output_layer = tf.keras.layers.Dense(
            num_actions, dtype=tf.float32)

    @tf.function
    def call(self, inputs):
        z = self.hidden_layer(inputs)
        output = self.output_layer(z)
        return output
        
class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self, t, num_actions):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(64, activation="relu")
    #self.dense1 = tf.keras.layers.Dense(32, activation="relu")
    #self.dense2 = tf.keras.layers.Dense(32, activation="relu")
    self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
  @tf.function 
  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    #x = self.dense2(x)
    return self.dense3(x)
import tensorflow as tf
        
class QNetwork(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self, t, num_actions):
    super(QNetwork, self).__init__()
    self.dense1 = tf.keras.layers.Dense(32, activation="relu")
    self.dense2 = tf.keras.layers.Dense(32, activation="relu")
    self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
    
  @tf.function 
  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    x = self.dense2(x)
    return self.dense3(x)
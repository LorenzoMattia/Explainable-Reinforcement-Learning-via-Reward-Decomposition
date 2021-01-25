import tensorflow as tf
import numpy as np
import lunar_lander
from lunar_lander import LunarLander

env = LunarLander()
nactions = env.action_space.n
gamma = 0.99

class MyModel(tf.keras.Model):
    def __init__(self, state_shape, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=(8,)
        )
        self.hidden_layer = tf.keras.layers.Dense(
            hidden_units, activation='relu', kernel_initializer='RandomNormal')
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        z = self.hidden_layer(z)
        output = self.output_layer(z)
        return output

# Output is three values, Q for all possible actions
Qs = MyModel(8, 64, nactions)

# Optimizer adjusts weights to minimize loss, with the speed of learning_rate
optimizer = tf.optimizers.Adam(1e-3)

epsilon = 0.1
max_episodes = 200
max_episode_length = 2000
eye = np.eye(8)

def predict(state):
    print("-------------------")
    print(np.shape(state))
    print("-------------------")
    return Qs(state)

def policy(eps, state):
    vals = predict(state)
    #nostra policy
    a = np.argmax(vals, axis=-1) if np.random.rand() >= eps  else np.random.randint(nactions, size=vals.shape[0])
    return a

for ep in range(max_episodes):
    if (ep) % 10 == 0:
        print(f'Episode {ep}')
        
    current_state = env.reset()
    d = False

    while not d:      
        a = policy(epsilon, current_state)
        o, r, d, _ = env.step(a[0])
        next_state = preprocess_state(o)

        rew = np.sum(s)
        
        with tf.GradientTape() as tape:
          selected_action_values = tf.math.reduce_sum(        # Q(s, a)
              predict(current_state) * tf.one_hot(a, nactions), axis=1)
          target = tf.stop_gradient(rew + (1 - d)*tf.reduce_max(predict(next_state), axis=1))  # r + (1 - d)*max(Q(s', a))
          loss = tf.math.reduce_mean(tf.square(target - selected_action_values))
        variables = Qs.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        current_state = next_state

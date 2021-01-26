import tensorflow as tf
import numpy as np
import lunar_lander
from lunar_lander import LunarLander

env = LunarLander()
num_actions = env.action_space.n
num_features = env.observation_space.shape[0]
print('Number of state features: {}'.format(num_features))
print('Number of possible actions: {}'.format(num_actions))


gamma = 0.99

class MyModel(tf.keras.Model):
    def __init__(self, state_shape, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=state_shape
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
Qs = MyModel(num_features, 64, num_actions)

# Optimizer adjusts weights to minimize loss, with the speed of learning_rate
optimizer = tf.optimizers.Adam(1e-3)

epsilon = 0.1
discount = 0.99
max_episodes = 1000
max_episode_length = 2000
eye = np.eye(8)

def demo_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = np.zeros(8)
    steps = 0
    s = env.reset()
    while True:
        a = policy(0, s)
        s, r, done, info = env.step(a[0])
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {}".format(steps, total_reward))
        steps += 1
        if done: break
    return total_reward

def predict(state):
    state_in = tf.expand_dims(state, axis=0)
    return Qs(state_in)

def policy(eps, state):
    vals = predict(state)
    #nostra policy
    a = np.argmax(vals, axis=-1) if np.random.rand() >= eps  else np.random.randint(num_actions, size=vals.shape[0])
    return a

for ep in range(max_episodes):
    if (ep) % 10 == 0:
        print(f'Episode {ep}')
        
    current_state = env.reset()
    d = False

    while not d:      
        a = policy(epsilon, current_state)    
        o, r, d, _ = env.step(a[0])
        next_state = o

        rew = np.sum(r)
        
        with tf.GradientTape() as tape:
          
          selected_action_values = tf.math.reduce_sum(        # Q(s, a)
              predict(current_state) * tf.one_hot(a, num_actions), axis=1)
          
          target = tf.stop_gradient(rew + (1 - d)*discount*tf.reduce_max(predict(next_state), axis=1))  # r + (1 - d)*γ*max(Q(s', a))
          loss = tf.math.reduce_mean(tf.square(target - selected_action_values))
        variables = Qs.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        current_state = next_state

demo_lander(env, render=True)
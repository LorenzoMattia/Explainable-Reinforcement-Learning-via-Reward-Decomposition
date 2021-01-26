import tensorflow as tf
import numpy as np
import lunar_lander
from lunar_lander import LunarLander
from collections import deque

env = LunarLander()
num_actions = env.action_space.n
num_features = env.observation_space.shape[0]
NUM_COMPONENT = 8
print('Number of state features: {}'.format(num_features))
print('Number of possible actions: {}'.format(num_actions))


gamma = 0.99

# BUILD NEURAL NETWORK

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


Qs = []
optimizer = []
for c in range(NUM_COMPONENT):
    Qs.append(MyModel(num_features, 64, num_actions))
    #Qs[c] = MyModel(num_features, 64, num_actions)
    optimizer.append(tf.optimizers.Adam(0.01))
    #optimizer[c] = tf.optimizers.Adam(0.01)


class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(np.array(state, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    dones = np.array(dones, dtype=np.float32)
    return states, actions, rewards, next_states, dones

epsilon = 0.1
discount = 0.99
max_episodes = 1000
max_episode_length = 2000
eye = np.eye(8)
buffer = ReplayBuffer(100000)
batch_size = 32

#Per stampare alla fine
def demo_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = np.zeros(8)
    steps = 0
    s = env.reset()
    while True:
        a = policy(s)
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

def predict(state, component):
    return Qs[component](state)
'''
def policy(eps, state):
    vals = predict(state)
    #nostra policy
    a = np.argmax(vals, axis=-1) if np.random.rand() >= eps  else np.random.randint(num_actions, size=vals.shape[0])
    return a
'''

def policy(state, eps=0):
    vals = np.zeros(num_actions)
    state_in = tf.expand_dims(state, axis=0)
    for c in range(NUM_COMPONENT):
        vals += predict(state_in,c)
    a = np.argmax(vals, axis=-1) if np.random.rand() >= eps  else np.random.randint(num_actions, size=vals.shape[0])
    return a

def train_step(states, actions, rewards, next_states, dones):
    for c in range(NUM_COMPONENT):
        with tf.GradientTape() as tape:      
          selected_action_values = tf.math.reduce_sum(        # Q(s, a)
              predict(states,c) * tf.one_hot(actions[:,0], num_actions), axis=-1)
          '''
          print("--------reward---------")
          print(np.shape(rewards))
          print(rewards)
          print("----------------------")
          '''
          target = tf.stop_gradient(rewards[:,c] + (1 - dones)*discount*tf.reduce_max(predict(next_states,c), axis=-1))  # r + (1 - d)*γ*max(Q(s', a))
          '''
          print("--------Target---------")
          print(np.shape(target))
          print(target)
          print("++++++++SelectActionValues+++++++++")
          print(np.shape(selected_action_values))
          print(selected_action_values)
          print("----------------------")
          '''
          loss = tf.math.reduce_sum(tf.square(target - selected_action_values))
          '''
          print("--------Loss---------")
          print(np.shape(loss))
          print(loss)
          '''
        
        variables = Qs[c].trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer[c].apply_gradients(zip(gradients, variables))


for ep in range(max_episodes):
    #if (ep) % 10 == 0:
        #print(f'Episode {ep}')
        
    current_state = env.reset()
    d = False
    ep_rew = np.zeros(NUM_COMPONENT)
    t = 0
    
    while not d:      
        a = policy(current_state, epsilon)    
        o, r, d, _ = env.step(a[0])
        next_state = o
        ep_rew += r
        
        buffer.add(current_state, a, r, next_state, d)
        
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            train_step(states, actions, rewards, next_states, dones)
            buffer = ReplayBuffer(100000)

        current_state = next_state
        t += 1
    
    print(f'Episode {ep}:/t Reward: {ep_rew}')
    
    
    
demo_lander(env, render=True)
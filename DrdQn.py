import gym
from collections import deque
import tensorflow as tf
import numpy as np
import random
import math
import lunar_lander
from lunar_lander import LunarLander
from GraphCollector import GraphCollector
import os, sys

env = LunarLander()
num_features = env.observation_space.shape[0]
num_actions = env.action_space.n
NUM_COMPONENT = env.num_reward_components
rew_plotter = GraphCollector()
loss_plotter = GraphCollector()
execute_policy = False
filepath = os.path.abspath(os.path.dirname(sys.argv[0])) + '\\Weights\\'
print('Number of state features: {}'.format(num_features))
print('Number of possible actions: {}'.format(num_actions))
max_episode_length = 1000


class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(32, activation="relu")
    self.dense2 = tf.keras.layers.Dense(32, activation="relu")
    self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
    
  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    x = self.dense2(x)
    return self.dense3(x)

def demo_lander(seed=None, render=False, prints=True):
    env.seed(seed)
    total_reward = np.zeros(NUM_COMPONENT)
    steps = 0
    s = env.reset()
    while True:
        s = tf.expand_dims(s, axis=0)
        a = select_epsilon_greedy_action(s, 0.)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if (steps % 20 == 0 or done) and prints:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {}".format(steps, total_reward))
        steps += 1
        if done: break
        if steps>max_episode_length: break
    return total_reward

def execute_some_policy(num_iterations = 10, seed=None, render=False, prints=False):
    total_reward = 0
    for i in range(num_iterations):
        total_reward += np.sum(demo_lander(seed,render,prints))
    avarage_reward = total_reward/num_iterations
    return avarage_reward

main_nn = []
target_nn = []
optimizer = []
for i in range(NUM_COMPONENT):
    main_nn.append(DQN())
    target_nn.append(DQN())
    optimizer.append(tf.keras.optimizers.Adam(1e-4))
mse = tf.keras.losses.MeanSquaredError()

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

def save_weights(overwrite=False):
    for c in range(NUM_COMPONENT):
        main_nn[c].save_weights(filepath + str(c)+'\\', overwrite=overwrite)   

def load_weights():
    for c in range(NUM_COMPONENT):
        main_nn[c].load_weights(filepath + str(c)+'\\')
    
def select_epsilon_greedy_action(state, epsilon):
  """Take random action with probability epsilon, else take best action."""
  result = tf.random.uniform((1,))
  if result < epsilon:
    return env.action_space.sample() # Random action (left or right).
  else:
    vals = np.zeros(num_actions)
    for i in range(NUM_COMPONENT):
        vals += (main_nn[i](state)[0])
    return tf.argmax(vals).numpy() # Greedy action for state.

@tf.function
def train_step(states, actions, rewards, next_states, dones):
  """Perform a training iteration on a batch of data sampled from the experience
  replay buffer."""
  # Calculate targets.
  totaloss = 0
  vals = np.zeros((batch_size,num_actions))
  for c in range(NUM_COMPONENT):
    vals += main_nn[c](next_states)
  next_actions = tf.math.argmax(vals, axis=-1)
  
  for c in range(NUM_COMPONENT):
    next_qs = target_nn[c](next_states)
    max_next_qs = tf.reduce_sum(next_qs * tf.one_hot(next_actions, num_actions), axis=-1)
    target = rewards[:,c] + (1. - dones) * discount * max_next_qs
    with tf.GradientTape() as tape:
        qs = main_nn[c](states)
        action_masks = tf.one_hot(actions, num_actions)
        masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
        loss = mse(target, masked_qs)
    totaloss += loss
    grads = tape.gradient(loss, main_nn[c].trainable_variables)
    optimizer[c].apply_gradients(zip(grads, main_nn[c].trainable_variables))
  return totaloss/NUM_COMPONENT

num_episodes = 25
batch_size = 32
discount = 0.99
buffer = ReplayBuffer(100000)

def train():
    cur_frame = 0
    epsilon = 0.9
    # Start training. Play game once and then train with a batch.
    last_100_ep_rewards = []
    for episode in range(num_episodes+1):
      t=0
      state = env.reset()
      ep_reward, done = 0, False
      ep_reward_composta = np.zeros(NUM_COMPONENT)
      totalLoss = []
      while not done:
        if t>max_episode_length: break
        t += 1
        state_in = tf.expand_dims(state, axis=0)
        action = select_epsilon_greedy_action(state_in, epsilon)
        next_state, reward, done, info = env.step(action)
        ep_reward += np.sum(reward)
        ep_reward_composta += reward
        # Save to experience replay.
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        cur_frame += 1
        # Copy main_nn weights to target_nn.
        if cur_frame % 2000 == 0:
          for c in range(NUM_COMPONENT):
            target_nn[c].set_weights(main_nn[c].get_weights())

        # Train neural network.
        if len(buffer) >= batch_size:
          states, actions, rewards, next_states, dones = buffer.sample(batch_size)
          loss = train_step(states, actions, rewards, next_states, dones)
          totalLoss.append(loss)
        
      #if episode < 950:
      if epsilon > 0.1:
        epsilon -= 0.8/900

      if len(last_100_ep_rewards) == 100:
        last_100_ep_rewards = last_100_ep_rewards[1:]
      last_100_ep_rewards.append(ep_reward)
      
      if (episode) % 25 == 0:
        tent_rew = execute_some_policy(10)
        rew_plotter.append(episode, tent_rew)
        meanLoss = np.mean(totalLoss)
        loss_plotter.append(episode, meanLoss)
        print(f'Episode :{episode}  Rew: {tent_rew:.3f}  MeanLoss: {meanLoss:.3f}  Steps: {t}')
      
      if episode % 50 == 0:
        print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
              f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards, axis=0):.3f}. '
              f'MeanLoss: {np.mean(totalLoss):.3f}')
              
      if episode % 100 == 0:
        print('Rew Scoposta: [', end='')
        for c in range(NUM_COMPONENT):
            print(f'{ep_reward_composta[c]:.2f}', end=', ')
        print(f"] Tot: {ep_reward:.3f}")
    env.close()
    if not filepath is None:
        save_weights(True)
    print(rew_plotter)
    rew_plotter.plot("Episodes", "Average Score")
    print(loss_plotter)
    loss_plotter.plot("Episodes", "Loss")
    demo_lander(render= True)

if __name__ == '__main__':

    if execute_policy :
        if not filepath is None:
            load_weights()
        else:
            print("Missing filepath")
        demo_lander(render = True)
    else:
        train()

import gym
from collections import deque
import tensorflow as tf
import numpy as np
import random
import math
from GraphCollector import GraphCollector

env = gym.make('LunarLander-v2')
num_features = env.observation_space.shape[0]
num_actions = env.action_space.n
rew_plotter = GraphCollector()
loss_plotter = GraphCollector()
print('Number of state features: {}'.format(num_features))
print('Number of possible actions: {}'.format(num_actions))

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
    total_reward = 0
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
    return total_reward

def execute_some_policy(num_iterations = 10, seed=None, render=False, prints=False):
    total_reward = 0
    for i in range(num_iterations):
        total_reward += demo_lander(seed,render,prints)
    avarage_reward = total_reward/num_iterations
    return avarage_reward


main_nn = DQN()
target_nn = DQN()

optimizer = tf.keras.optimizers.Adam(1e-4)
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


def select_epsilon_greedy_action(state, epsilon):
  """Take random action with probability epsilon, else take best action."""
  result = tf.random.uniform((1,))
  if result < epsilon:
    return env.action_space.sample() # Random action (left or right).
  else:
    return tf.argmax(main_nn(state)[0]).numpy() # Greedy action for state.


@tf.function
def train_step(states, actions, rewards, next_states, dones):
  """Perform a training iteration on a batch of data sampled from the experience
  replay buffer."""
  # Calculate targets.
  next_qs = target_nn(next_states)
  max_next_qs = tf.reduce_max(next_qs, axis=-1)
  target = rewards + (1. - dones) * discount * max_next_qs
  with tf.GradientTape() as tape:
    qs = main_nn(states)
    action_masks = tf.one_hot(actions, num_actions)
    masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
    loss = mse(target, masked_qs)
  grads = tape.gradient(loss, main_nn.trainable_variables)
  optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
  return loss

num_episodes = 1500
epsilon = 1.0
batch_size = 32
discount = 0.99
buffer = ReplayBuffer(100000)
cur_frame = 0

# Start training. Play game once and then train with a batch.
last_100_ep_rewards = []
for episode in range(num_episodes+1):
  t=0
  state = env.reset()
  ep_reward, done = 0, False
  meanLoss = []
  while not done:
    t += 1
    state_in = tf.expand_dims(state, axis=0)
    action = select_epsilon_greedy_action(state_in, epsilon)
    next_state, reward, done, info = env.step(action)
    ep_reward += reward
    # Save to experience replay.
    buffer.add(state, action, reward, next_state, done)
    state = next_state
    cur_frame += 1
    # Copy main_nn weights to target_nn.
    if cur_frame % 2000 == 0:
      target_nn.set_weights(main_nn.get_weights())

    # Train neural network.
    if len(buffer) >= batch_size:
      states, actions, rewards, next_states, dones = buffer.sample(batch_size)
      loss = train_step(states, actions, rewards, next_states, dones)
      meanLoss.append(loss)
  
  if episode < 950:
    epsilon -= 0.001

  if len(last_100_ep_rewards) == 100:
    last_100_ep_rewards = last_100_ep_rewards[1:]
  last_100_ep_rewards.append(ep_reward)
  
  if (episode) % 25 == 0:
    tent_rew = execute_some_policy(10)
    rew_plotter.append(episode, tent_rew)
    meanLoss = np.mean(meanLoss)
    loss_plotter.append(episode, meanLoss)
    print(f'Episode :{episode}  Rew: {tent_rew}  MeanLoss: {meanLoss}')
    
  if episode % 50 == 0:
    print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
          f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}. '
          f'MeanLoss: {np.mean(meanLoss):.3f}')
env.close()
print(rew_plotter)
rew_plotter.plot("Episodes", "Average Score")
print(loss_plotter)
loss_plotter.plot("Episodes", "Loss")


env = gym.make('LunarLander-v2')
state = env.reset()
done = False
ep_rew = 0
while not done:
  env.render()
  state = tf.expand_dims(state, axis=0)
  action = select_epsilon_greedy_action(state, epsilon=0.01)
  state, reward, done, info = env.step(action)
  ep_rew += reward
print('Episode reward was {}'.format(ep_rew))
env.close()


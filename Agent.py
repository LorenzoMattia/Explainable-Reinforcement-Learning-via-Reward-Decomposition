import tensorflow as tf
import numpy as np
import lunar_lander
from lunar_lander import LunarLander
from GraphCollector import GraphCollector
from ReplayBuffer import ReplayBuffer
from QNetwork import QNetwork
import os, sys
from cliff_world import CliffWalkingEnv
import gym
import random
from bisect import bisect
from heapq import nsmallest
import itertools
from Explainer import Explainer
import math


class Agent():

    """
    A class used to represent a Reinforcement Learning agent implementing QLearning algorithm in a decomposed reward setting adaptable to any environment which is provided of:
    1) decomposed reward
    2) a variable num_reward_components representing the number of components of the reward

    Attributes:
    -----------------------------------------------------------------------------------------
    env : Gym.Env
        the gym environment modified as described above
    QNetwork : tf.keras.Model
        the model predicting the Q values through the use of the call function
    epsilon : float
        hyper parameter expressing the agent initial probability of taking a random action instead of the one learned and stored in the policy
    exploration_type : int
        0 eps-Boltzmann exploration, 1 eps-greedy exploration, 2 Boltzmann exploration
    discount : float
        hyper parameter reducing Q value of future state-action couples
    max_episodes : int
        maximum number of episodes
    max_episode_length : int
        maximum number of step of an episode
    batch_size : int
        number of states on which the network is trained in a single shot
    discount_decay_episodes : int
        number of episodes necessary for decreasing the discount factor from 0.9 to 0.1
    plot_point : int
        variable expressing every how many episodes the agent evaluates the policy for plotting results
    num_policy_exe : int
        number of times the policy is executed (every plot_point episodes) for averaging results and plot graphs
    continue_learning : Bool
        True if you want to continue the learning process starting from weights already learned False otherwise
    execute_policy : int
        0 if you want to train, 1 if you want to execute the learned policy, 2 if you want to execute the policy and get the explanation.
    """

    def __init__(self, env = LunarLander(), QNet = QNetwork, exploration_type = 0, epsilon = 0.9, discount = 0.99, max_episodes = 1000, max_episode_length = 1000,
                batch_size = 32, discount_decay_episodes = 300, plot_point = 25, num_policy_exe = 10, continue_learning = False,
                execute_policy = 0, filepath = os.path.abspath(os.path.dirname(sys.argv[0])) + '\\Weights\\'):

        self.env = env
        self.exploration_type = exploration_type
        self.epsilon = epsilon if not exploration_type==2 else 2
        self.phi = 100 if exploration_type==2 else 1
        self.discount = discount
        self.max_episodes = max_episodes
        self.max_episode_length = max_episode_length
        self.batch_size = batch_size
        self.continue_learning = continue_learning
        self.execute_policy = execute_policy
        self.buffer = ReplayBuffer(100000)
        self.rew_plotter = GraphCollector()
        self.loss_plotter = GraphCollector()
        self.num_actions = self.env.action_space.n
        self.NUM_COMPONENT = self.env.num_reward_components
        self.discount_decay_episodes = discount_decay_episodes
        self.plot_point = plot_point
        self.main_nn = []
        self.target_nn = []
        self.optimizer = []
        self.mse = tf.keras.losses.MeanSquaredError()
        self.num_policy_exe = num_policy_exe
        self.filepath = filepath
        self.explainer = Explainer(self)
        
        for c in range(self.NUM_COMPONENT):
            self.main_nn.append(QNet(64, self.num_actions))
            self.target_nn.append(QNet(64, self.num_actions))
            self.optimizer.append(tf.optimizers.Adam(0.01)) #5e-4

    def train(self):
        cur_frame = 0
        if self.continue_learning:
            if not self.filepath is None:
                self.load_weights()
            else:
                print("Missing filepath")
                return
        last_100_ep_rewards = []
        for ep in range(self.max_episodes+1):

            current_state = self.reset()
            d = False
            ep_rew = 0
            step = 0
            totalLoss = []
            while not d:
                if step > self.max_episode_length: break
                step += 1
                a = self.policy(current_state, self.epsilon)
                o, r, d, _ = self.step(a)
                next_state = o
                ep_rew += np.sum(r)

                self.buffer.add(current_state, a, r, next_state, d)
                current_state = next_state
                cur_frame += 1
                if cur_frame % 2000 == 0:
                    #Update target's weights
                    self.update_target_weights()

                if len(self.buffer) >= self.batch_size:
                    states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                    loss = self.train_step(states, actions, rewards, next_states, dones)
                    totalLoss.append(loss)

            #Exploration Decay
            if self.exploration_type == 2:               #Boltzmann
                if self.epsilon > -0.299:
                    self.epsilon -= 1.8/self.discount_decay_episodes  
                    self.phi = math.pow(10,self.epsilon)            
            else:                                       #Other cases
                if self.epsilon > 0.1:
                    self.epsilon -= 0.8/self.discount_decay_episodes
                
            if len(last_100_ep_rewards) == 100:
                last_100_ep_rewards = last_100_ep_rewards[1:]
            last_100_ep_rewards.append(ep_rew)

            if (ep) % self.plot_point == 0:
                tent_rew = self.execute_some_policy()
                self.rew_plotter.append(ep, tent_rew)
                meanLoss = np.mean(totalLoss)
                self.loss_plotter.append(ep, meanLoss)
                print(f'Episode :{ep}  Rew: {tent_rew}  MeanLoss: {meanLoss}    BufferSize: {len(self.buffer)}')

            if ep % 50 == 0:
                print(f'Episode {ep}/{self.max_episodes}. Epsilon: {self.epsilon:.3f}. '
                      f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards, axis=0):.3f}. '
                      f'MeanLoss: {np.mean(totalLoss):.3f}')

        if not self.filepath is None:
            self.save_weights(True)
        print(self.rew_plotter)
        self.rew_plotter.plot("Episodes", "Average Score")
        print(self.loss_plotter)
        self.loss_plotter.plot("Episodes", "Loss")
        self.demo_lander(render = True)

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        totaloss = 0
        vals = np.zeros((self.batch_size,self.num_actions))
        #computation of the best actions to perform starting from all the states contained in the batch-> the computation is based on the prediction made by the main nns
        for c in range(self.NUM_COMPONENT):
            vals += self.predict(next_states,c)  #summation of all the components predictions
        next_actions = tf.math.argmax(vals, axis=-1)

        for c in range(self.NUM_COMPONENT):
            next_qs = self.target_nn[c](next_states)    #Q value predicted by the target nns for the next states for a specific component
            max_next_qs = tf.reduce_sum(next_qs* tf.one_hot(next_actions, self.num_actions), axis=-1)   #selection of the best Qvalue predicted for that component
            target = rewards[:,c] + (1. - dones) * self.discount * max_next_qs  #Qvalues for the target nns, they are more precise because of the summation with the current reward

            with tf.GradientTape() as tape:
                selected_action_values = tf.reduce_sum(        # Q(s, a)
                  self.predict(states,c) * tf.one_hot(actions, self.num_actions), axis=-1)  #Qvalue prediction made by the main nns
                loss = self.mse(target, selected_action_values)
            totaloss += loss
            gradients = tape.gradient(loss, self.main_nn[c].trainable_variables)
            self.optimizer[c].apply_gradients(zip(gradients, self.main_nn[c].trainable_variables))
        mean_loss = totaloss/self.NUM_COMPONENT
        return mean_loss
    
    
    def policy_explanation(self, render = False, seed = None):
        self.env.seed(seed)
        steps = 0
        total_reward = np.zeros(self.NUM_COMPONENT)
        s = self.reset()
        rdxlist = []
        chosenactions = []
        done = False
        while not done:
            a = self.policy(s)
            chosenactions.append(a)
            s, r, done, info = self.step(a)
            total_reward += r
            rdxtot = np.zeros(((self.num_actions-1), self.NUM_COMPONENT))
            for i in range(self.num_actions):
                if not i == a:
                    j = i -1 if i > a else i
                    rdxtot[j, :] = self.explainer.compute_rdx(s, a, i)   #summation of rdx between all the actions in one step
            rdxlist.append(rdxtot)
            if render:
                still_open = self.env.render()
                if still_open == False: break
            steps += 1
            if steps>self.max_episode_length: break 
        
        self.explainer.compute_explanation(rdxlist,chosenactions)
        
        
            
    def demo_lander(self, seed=None, render=False, prints=True):
        self.env.seed(seed)
        total_reward = np.zeros(self.NUM_COMPONENT)
        steps = 0
        s = self.reset()
        while True:
            a = self.policy(s)
            s, r, done, info = self.step(a)
            total_reward += r

            if render:
                still_open = self.env.render()
                if still_open == False: break

            if (steps % 20 == 0 or done) and prints:
                print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
                print("step {} total_reward {}".format(steps, total_reward))
            steps += 1

            if done: break
            if steps>self.max_episode_length: break
        return total_reward


    def save_weights(self, overwrite=False):
        for c in range(self.NUM_COMPONENT):
            self.main_nn[c].save_weights(self.filepath + str(c)+'\\', overwrite=overwrite)

    def load_weights(self):
        for c in range(self.NUM_COMPONENT):
            self.main_nn[c].load_weights(self.filepath + str(c)+'\\')

    def update_target_weights(self):
        for c in range(self.NUM_COMPONENT):
            self.target_nn[c].set_weights(self.main_nn[c].get_weights())

    def policy(self, state, eps=0):
        #Compute Q-value
        vals = np.zeros(self.num_actions)
        state_in = tf.expand_dims(state, axis=0)
        for c in range(self.NUM_COMPONENT):
            vals += self.predict(state_in,c)[0]

        if self.exploration_type == 2:
            return self.Boltzmann_action(vals)          #Boltzmann
        
        rand = tf.random.uniform((1,))
        if rand<eps:
            if self.exploration_type == 1:
                return self.env.action_space.sample()   #eps-greedy
            return self.Boltzmann_action(vals)          #eps-Boltzmann
        
        return tf.argmax(vals).numpy()                  #Exploitation
    
    def Boltzmann_action(self, vals):
        exp_vals = [math.exp(x/self.phi) for x in vals]
        tot_exp_vals = np.sum(exp_vals)
        exp_vals = exp_vals/tot_exp_vals
        cdf = [exp_vals[0]]
        for i in range(1, len(exp_vals)):
            cdf.append(cdf[-1] + exp_vals[i])
        random_ind = bisect(cdf,random.random())
        return random_ind
     
    def execute_some_policy(self, seed=None, render=False, prints=False):
        total_reward = 0
        for i in range(self.num_policy_exe):
            total_reward += np.sum(self.demo_lander(seed,render,prints))
        avarage_reward = total_reward/self.num_policy_exe
        return avarage_reward

    def predict(self, state, component):
        return self.main_nn[component](state)

    def main(self):
        if self.execute_policy == 0:
            self.train()
            return
        if not self.filepath is None:
            self.load_weights()
        else:
            print("Missing filepath")
            return
        if self.execute_policy == 1:
            self.demo_lander(render=True, prints=False)
            return
        if self.execute_policy == 2:
            self.policy_explanation(render = True)
            return
        print("execute_policy invalid value")
        return

    def step(self, action):
        o, r, d, info = self.env.step(action)
        if np.shape(o) == ():
            o = tf.one_hot(o, 20)
        return o, r, d, info

    def reset(self):
        o = self.env.reset()
        if np.shape(o) == ():
            o = tf.one_hot(o, 20)
        return o
    

if __name__ == '__main__':

    agent = Agent(execute_policy = 2)

    agent.main()

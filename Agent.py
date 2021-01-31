import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import lunar_lander
from lunar_lander import LunarLander
from GraphCollector import GraphCollector
from ReplayBuffer import ReplayBuffer
from QNetwork import QNetwork
import os, sys
from cliff_world import CliffWalkingEnv

#filepath = 'C:\\Users\\Lorenzo\\Documents\\Didattica Uni\\ArtificialIntelligenceRobotics\\Primo anno\\ReinforcementLearning\\ProgettoEsame\\Explainable-Reinforcement-Learning-via-Reward-Decomposition\\Weights\\'
#filepath = 'D:\\Users\\norto\\Documents\\Università La Sapienza\\RL\\Progetto_Esame\\Explainable-Reinforcement-Learning-via-Reward-Decomposition\\Weights\\'

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
    execute_policy : Bool
        True if you want to execute the learned policy instead of continue learning False otherwise
    """

    def __init__(self, env = CliffWalkingEnv(), QNetwork = QNetwork, epsilon = 0.9, discount = 0.99, max_episodes = 500, max_episode_length = 3000,
                batch_size = 32, discount_decay_episodes = 400, plot_point = 25, num_policy_exe = 10, continue_learning = False, 
                execute_policy = False, filepath = os.path.abspath(os.path.dirname(sys.argv[0])) + '\\Weights\\'):
        
        self.env = env
        self.epsilon = epsilon
        self.discount = discount
        self.max_episodes = max_episodes
        self.max_episode_length = max_episode_length
        self.batch_size = batch_size
        self.continue_learning = continue_learning
        self.execute_policy = execute_policy
        self.buffer = ReplayBuffer(1000000)
        self.plotter = GraphCollector()
        self.num_actions = self.env.action_space.n
        self.NUM_COMPONENT = self.env.num_reward_components
        self.discount_decay_episodes = discount_decay_episodes
        self.plot_point = plot_point
        self.main_nn = []
        self.target_nn = []
        self.optimizer = []
        self.num_policy_exe = num_policy_exe
        self.filepath = filepath
        for c in range(self.NUM_COMPONENT):
            self.main_nn.append(QNetwork(64, self.num_actions))
            self.target_nn.append(QNetwork(64, self.num_actions))
            self.optimizer.append(tf.optimizers.Adam(0.01))
    
    def train(self):
        cur_frame = 0
        
        if self.continue_learning:
            if not self.filepath is None:
                self.load_weights()
            else:
                print("Missing filepath")
                return
        self.update_target_weights()

        for ep in range(self.max_episodes):
            
            current_state = self.reset()
            d = False
            ep_rew = np.zeros(self.NUM_COMPONENT)
            step = 0
            meanLoss = 0
            num_train = 0
            while not d:
                if step > self.max_episode_length:
                    break
                a = self.policy(current_state, self.epsilon)    
                o, r, d, _ = self.step(a[0])
                next_state = o
                ep_rew += r
                
                self.buffer.add(current_state, a, r, next_state, d)
                cur_frame += 1
                if cur_frame % 2000 == 0:
                    #Update target's weights
                    self.update_target_weights()
                
                if len(self.buffer) >= self.batch_size:
                    states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                    meanLoss += self.train_step(states, actions, rewards, next_states, dones)
                    num_train += 1
                current_state = next_state
                
                step += 1
                
            if self.epsilon > 0.1:
                self.epsilon -= 0.8/self.discount_decay_episodes
            
            
            if (ep) % self.plot_point == 0:
                tent_rew = self.execute_some_policy(10)
                self.plotter.append(ep, tent_rew)
                meanLoss = meanLoss/num_train if not num_train == 0 else 'not computed'
                print(f'Episode :{ep}  Rew: {tent_rew}  MeanLoss: {meanLoss}')
            
            print(f'Episode :{ep}   Step :{step}   Rew: {ep_rew}')
        
        if not self.filepath is None:
            self.save_weights(True)
        print(self.plotter)
        self.plotter.plot("Episodes", "Average Score")
        self.demo_lander(render = True)
    
    
    def train_step(self, states, actions, rewards, next_states, dones):
        totaloss = 0
        vals = np.zeros((self.batch_size,self.num_actions))

        for c in range(self.NUM_COMPONENT):
            vals += self.predict(next_states,c)
        next_actions = np.argmax(vals, axis=-1)
        
        for c in range(self.NUM_COMPONENT):
            next_qs = self.target_nn[c](next_states)
            max_next_qs = tf.math.reduce_sum(next_qs* tf.one_hot(next_actions, self.num_actions), axis=-1)
            target = rewards[:,c] + (1. - dones) * self.discount * max_next_qs
            
            with tf.GradientTape() as tape:
                selected_action_values = tf.math.reduce_sum(        # Q(s, a)
                  self.predict(states,c) * tf.one_hot(actions[:,0], self.num_actions), axis=-1)
                
                temp = tf.square(target - selected_action_values)
                loss = tf.math.reduce_mean(temp)
                
            totaloss +=loss
            variables = self.main_nn[c].trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer[c].apply_gradients(zip(gradients, variables))
        return loss/self.NUM_COMPONENT
        
        
    def demo_lander(self, seed=None, render=False, prints=True):
        self.env.seed(seed)
        total_reward = np.zeros(self.NUM_COMPONENT)
        steps = 0
        s = self.reset()
        while True:
            a = self.policy(s)
            s, r, done, info = self.step(a[0])
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
        vals = np.zeros(self.num_actions)
        state_in = tf.expand_dims(state, axis=0)
        for c in range(self.NUM_COMPONENT):
            vals += self.predict(state_in,c)
            
        temp = np.random.rand()
        a = np.argmax(vals, axis=-1) if  temp >= eps  else np.random.randint(self.num_actions, size=vals.shape[0])
        return a
        
    def execute_some_policy(self, num_iterations = 10, seed=None, render=False, prints=False):
        total_reward = 0
        for i in range(num_iterations):
            total_reward += np.sum(self.demo_lander(seed,render,prints))
        avarage_reward = total_reward/num_iterations
        return avarage_reward

    def predict(self, state, component):
        return self.main_nn[component](state)
        
    def main(self):
        if self.execute_policy :
            if not self.filepath is None:
                self.load_weights()
            else:
                print("Missing filepath")
                return
            self.demo_lander(env, render = True)
        else:
            self.train() 
                    
    def step(self, action):
        o, r, d, info = self.env.step(action)
        if np.shape(o) == ():
            o = np.array([o])
        return o, r, d, info
        
    def reset(self):
        o = self.env.reset()
        if np.shape(o) == ():
            o = np.array([o])
        return o

        
if __name__ == '__main__':
    agent = Agent()
    agent.main()
        
        
        
        
        
        
        
        
        
        
        
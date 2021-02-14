import numpy as np
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf

class Explainer():
    def __init__(self, agent):
        self.agent = agent
        self.num2actions = {0 : "noop", 1: "leftengine", 2: "mainengine", 3: "rightengine"} #only for Lunar Lander
        self.components_names = ['crash', 'live', 'main fuel cost', 'side fuel cost', 'angle', 'contact', 'distance', 'velocity'] #only for Lunar Lander
        
    def compute_msx(self, rdx, k, positive = True):
        sign = 1 if positive else -1
        msx = []
        for L in range(0, len(rdx)+1):
            allcomb = []
            for subset in itertools.combinations(sign*rdx, L):
                allcomb.append(subset)
            allcomb = np.array(allcomb)
            greaters = self.graterthan(allcomb, k)
            if greaters:
                msx = sign*min(greaters, key = sum)
                break
                
        return np.array(msx)
        
    def compute_rdx(self, state, a1, a2):
        state = tf.expand_dims(state, axis=0)
        qvals = self.agent.predict(state, 0)[0]
        for c in range(1, self.agent.NUM_COMPONENT):
            qvals = np.vstack((qvals,self.agent.predict(state, c)[0]))
        q1 = qvals[:, a1]
        q2 = qvals[:, a2]

        rdx = q1-q2     #ogni elemento del vettore risultante è l'rdx di una componente
        return rdx
        
    def compute_d(self, rdx):
        d = 0
        for i in range(len(rdx)):
            if rdx[i]<0:
                d += rdx[i]
        return abs(d)
        
    def compute_v(self, msxplus):
        return np.sum(msxplus) - msxplus.min() if not len(msxplus) == 0 else 0
        
    def explanation(self, state, a1, a2, plot = False):
    
        rdx = self.compute_rdx(state, a1, a2)
        values2names = dict(zip(rdx, self.components_names))
        d = self.compute_d(rdx)
        msxplus = self.compute_msx(rdx, d)
        v = self.compute_v(msxplus)
        msxmin = self.compute_msx(rdx, v, False)
        
        if plot:
            print(f'Action to do: {self.num2actions[a1]}, Action to compare: {self.num2actions[a2]}')
            print(f'RDX: {rdx}')
            print(f'msxplus = {msxplus}')
            print(f'v = {v}')
            print(f'msxmin = {msxmin}')
            #plot rdx
            x = np.arange(self.agent.NUM_COMPONENT)
            plt.bar(x, height = rdx)
            plt.xticks(x, ['crash', 'live', 'main fuel cost', 'side fuel cost', 'angle', 'contact', 'distance', 'velocity'], rotation = 45)
            plt.show()
            
            x = np.arange(len(msxplus))
            plt.title('MSX+ ' + self.num2actions[a1] + 'vs' + self.num2actions[a2])
            plt.bar(x, height = msxplus)
            plt.xticks(x, [values2names[x] for x in msxplus], rotation = 45)
            plt.show()
            
            x = np.arange(len(msxmin))
            plt.title('MSX- ' + self.num2actions[a1] + 'vs' + self.num2actions[a2])
            plt.bar(x, height = msxmin)
            plt.xticks(x, [values2names[x] for x in msxmin], rotation = 45)
            plt.show()
            print()
            
        return rdx
        
    def graterthan(self, lst, k):
        result = []
        for i in lst:
            if np.sum(i) > k:
                result.append(i)
        return result
        
        
    def computeallmsx(self, rdxlist, chosenactions = None): 
        component2timesinmsxplus = dict(zip(self.components_names, np.zeros(self.agent.NUM_COMPONENT)))
        component2timesinmsxmin = component2timesinmsxplus.copy()
        actions2componentsinmsxplus = {}
        actions2componentsinmsxmin = {}
        for i in range(self.agent.num_actions):
            actions2componentsinmsxplus[i] = component2timesinmsxplus.copy()
            actions2componentsinmsxmin[i] = component2timesinmsxplus.copy()
        #actions2componentsinmsxmin = actions2componentsinmsxplus.copy()
        for i,rdx in enumerate(rdxlist):
            values2names = dict(zip(rdx, self.components_names))
            d = self.compute_d(rdx)
            msxplus = self.compute_msx(rdx, d)
            v = self.compute_v(msxplus)
            msxmin = self.compute_msx(rdx, v, False)
            #print(msxplus)
            #print(msxmin)
            for c in msxplus:
                component2timesinmsxplus[values2names[c]] = component2timesinmsxplus[values2names[c]] +1
                if chosenactions is not None:
                    actions2componentsinmsxplus[chosenactions[i]][values2names[c]] = actions2componentsinmsxplus[chosenactions[i]][values2names[c]] + 1
            for c in msxmin:
                component2timesinmsxmin[values2names[c]] = component2timesinmsxmin[values2names[c]] +1
                if chosenactions is not None:
                    actions2componentsinmsxmin[chosenactions[i]][values2names[c]] = actions2componentsinmsxmin[chosenactions[i]][values2names[c]] + 1
        
        return component2timesinmsxplus, component2timesinmsxmin, actions2componentsinmsxplus, actions2componentsinmsxmin
     
    def msx_actionVSsaction(self, rdxlist, chosenactions):
        action2action2msxplus = {}
        action2action2msxmin = {}
        actions2rdx = {}
        for i in range(self.agent.num_actions):
            actions2rdx[i] = []
        for i, a in enumerate(chosenactions):
            actions2rdx[a].append(rdxlist[i])
        for i in actions2rdx.keys():
            if not actions2rdx[i]: continue 
            actionvs2msxplus = {}
            actionvs2msxmin = {}
            temp = np.array(actions2rdx[i])
            for j in range(self.agent.num_actions-1):
                actionvs = temp[:,j,:]
                plus, min, _, _ = self.computeallmsx(actionvs)
                k = j if j<i else j+1
                actionvs2msxplus[self.num2actions[k]] = plus
                actionvs2msxmin[self.num2actions[k]] = min
            action2action2msxplus[i] = actionvs2msxplus
            action2action2msxmin[i] = actionvs2msxmin
        return action2action2msxplus, action2action2msxmin
        
    def compute_state_explanation(self, state):
        a = self.agent.policy(state)
        rdxtot = np.zeros(((self.agent.num_actions -1),self.agent.NUM_COMPONENT))
        for i in range(self.agent.num_actions):
            if not i == a:
                j = i -1 if i > a else i
                rdxtot[j, :] = self.compute_rdx(state, a, i)
        rdxtot = np.expand_dims(rdxtot, axis=0)
        chosenaction = [a]
        action2action2msxplus, action2action2msxmin = self.msx_actionVSsaction(rdxtot, chosenaction)
        actionvs2msxplus = list(action2action2msxplus.values())[0]
        actionvs2msxmin = list(action2action2msxmin.values())[0]
        return rdxtot, actionvs2msxplus, actionvs2msxmin
    
    def compute_state_explanation2(self, state):
        a = self.agent.policy(state)
        for i in range(self.agent.num_actions):
            if not i == a:
                self.explanation(state, a, i, plot=True)
        
    def state_explanation(self):
        #Only for LunarLandar
        print(type(self.agent.env))
        num_comp_state = self.agent.env.observation_space.shape[0]
        s = np.zeros(num_comp_state)
        '''
        [ 1.5160218e-01 -1.2227717e-02 -1.6658909e-03  1.3573238e-04  -9.1273762e-02  5.8031554e-04  1.0000000e+00  1.0000000e+00]
        [ 1.51621729e-01, -1.22313285e-02,  2.01031147e-03, -1.61580349e-04,  -9.13095027e-02, -6.99558645e-04,  1.00000000e+00,  1.00000000e+00]
        [ 1.5160170e-01, -1.2227645e-02, -2.0603293e-03,  1.6453151e-04,  -9.1272883e-02,  7.1657787e-04,  1.0000000e+00,  1.0000000e+00]
        [ 1.5162019e-01, -1.2231043e-02,  1.9032208e-03, -1.5269163e-04,  -9.1306731e-02, -6.6219451e-04,  1.0000000e+00,  1.0000000e+00]
        [ 1.51607037e-01, -1.22284675e-02, -1.34081964e-03,  1.15886432e-04,  -9.12827477e-02,  4.69277700e-04,  1.00000000e+00,  1.00000000e+00]
        '''
        '''
        s[0] = 1.5160218e-01                #s[0] is the horizontal coordinate [-1,1]
        s[1] = -1.2227717e-02               #s[1] is the vertical coordinate [1.4,0]
        s[2] = -1.6658909e-03             #s[2] is the horizontal speed 
        s[3] = 1.3573238e-04             #s[3] is the vertical speed 
        s[4] = -9.1273762e-02              #s[4] is the angle 
        s[5] = 5.8031554e-04              #s[5] is the angular speed 
        s[6] = 1                #s[6] 1 if first leg has contact, else 0
        s[7] = 1                #s[7] 1 if second leg has contact, else 0 
        '''
        s = [ 1.51607037e-01, -1.22284675e-02, -1.34081964e-03,  1.15886432e-04,  -9.12827477e-02,  4.69277700e-04,  1.00000000e+00,  1.00000000e+00]
        rdxtot, actionvs2msxplus, actionvs2msxmin = self.compute_state_explanation(s)
        print(rdxtot)
        print(actionvs2msxplus)
        print(actionvs2msxmin)
        self.compute_state_explanation2(s)


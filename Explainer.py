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
        
    def explanation(state, a1, a2, plot = False):
    
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
            x = np.arange(NUM_COMPONENT)
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
                
                
    
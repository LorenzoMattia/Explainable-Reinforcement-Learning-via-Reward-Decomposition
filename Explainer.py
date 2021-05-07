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
    

    def compute_mnx(self, rdx, d):
        mnx = []
        rdx = rdx[rdx > 0]
        for L in range(0, len(rdx)+1):
            allcomb = []
            for subset in itertools.combinations(rdx, L):
                allcomb.append(subset)
            allcomb = np.array(allcomb)
            lowers = self.remaining_more_d(rdx, allcomb, d)
            if lowers:
                mnx = max(lowers, key = sum)
                break
        return mnx
    
    def remaining_more_d(self, rdx, allcomb, d):
        result = []
        i=0
        
        if d == 0:
            return []
        for lst in allcomb:
            i += 1
            temp = list(rdx)
            for element in lst:
                if element in temp:
                    temp.remove(element)
            if not temp or np.sum(temp)<d:
                result.append(lst)
            
        return result
    
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
        
    def explanation(self, state, a1, a2, print=False, plot=False):
    
        rdx = self.compute_rdx(state, a1, a2)
        values2names = dict(zip(rdx, self.components_names))
        d = self.compute_d(rdx)
        msxplus = self.compute_msx(rdx, d)
        v = self.compute_v(msxplus)
        msxmin = self.compute_msx(rdx, v, False)
        mnx = self.compute_mnx(rdx, d)
        if print:
            print(f'Action to do: {self.num2actions[a1]}, Action to compare: {self.num2actions[a2]}')
            print(f'RDX: {rdx}')
            print(f'msxplus = {msxplus}')
            print(f'v = {v}')
            print(f'msxmin = {msxmin}')
            print(f'mnx = {mnx}')
        if plot:
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
        
    

    def computeallmnx(self, rdxlist): 
        component2timesinmnx = dict(zip(self.components_names, np.zeros(self.agent.NUM_COMPONENT)))
        for i,rdx in enumerate(rdxlist):
            values2names = dict(zip(rdx, self.components_names))
            d = self.compute_d(rdx)
            mnx = self.compute_mnx(rdx, d)
            for c in mnx:
                component2timesinmnx[values2names[c]] = component2timesinmnx[values2names[c]] +1
        
        return component2timesinmnx
        
    def mnx_actionVSsaction(self, rdxlist, chosenactions):
        action2action2mnx = {}
        actions2rdx = {}
        for i in range(self.agent.num_actions):
            actions2rdx[i] = []
        for i, a in enumerate(chosenactions):
            actions2rdx[a].append(rdxlist[i])           #action2rdx associa ad ognuna delle 4 azioni tutti gli rdx rispetto alle altre azioni di tutti gli step
        for i in actions2rdx.keys():
            if not actions2rdx[i]: continue 
            actionvs2mnx = {}
            temp = np.array(actions2rdx[i])
            for j in range(self.agent.num_actions-1):
                actionvs = temp[:,j,:]
                mnx = self.computeallmnx(actionvs)
                k = j if j<i else j+1
                actionvs2mnx[self.num2actions[k]] = mnx
            action2action2mnx[i] = actionvs2mnx
        return action2action2mnx
       
       
    def compute_explanation(self, rdxlist, chosenactions):
        rdxlistmean = np.sum(rdxlist, axis = 1)
        
        print("\n\nResult of Explanation:\n")
        print(f"mean of the medium rdx of all the steps {np.mean(rdxlistmean, axis = 0)}\n")            #mean of the medium rdx of all the steps
        print(f"mean of the medium rdx of the first steps {np.mean(rdxlistmean[:50], axis = 0)}\n")
        print(f"mean of the medium rdx of the last steps {np.mean(rdxlistmean[150:], axis = 0)}\n")                       
        print(f"median of the medium rdx of all the steps {np.median(rdxlistmean, axis=0)}\n")           #median of the medium rdx of all the steps
        
        '''
        an element of rdxlistmean is the rdx of the chosen action compared to the average reward of the other actions;
        component2timesinmsx contains for each component the number of times it appeared in the msxplus,
        i.e. that choosing a certain action made it possible to overcome the disadvantage to be done also with msxmin
        '''
        
        component2timesinmsxplus, component2timesinmsxmin,  \
        actions2componentsinmsxplus, actions2componentsinmsxmin = self.computeallmsx(rdxlistmean, chosenactions)
        
        component2timesinmnx = self.computeallmnx(rdxlistmean)
        
        action2action2msxplus, action2action2msxmin = self.msx_actionVSsaction(rdxlist, chosenactions)
        action2action2mnx = self.mnx_actionVSsaction(rdxlist, chosenactions)
        
        print(f"Times per component in msx+ {component2timesinmsxplus}\n")
        print(f"Times per component in msx- {component2timesinmsxmin}\n")
        print(f"Times per component in mnx  {component2timesinmnx}\n")
        
        count_mnx = np.zeros(8)
        
        for i in range(self.agent.num_actions):
            print(f"action {self.num2actions[i]} executed {chosenactions.count(i)} times")
            print(f"msx+ : {actions2componentsinmsxplus[i]}")
            print(f"msx- : {actions2componentsinmsxmin[i]}\n")
            for j in action2action2msxplus[i].keys():
                print(f"{self.num2actions[i]} vs {j}: ")
                print(f"msx+ : {action2action2msxplus[i][j]}")
                print(f"msx- : {action2action2msxmin[i][j]}")
                print(f"mnx  : {action2action2mnx[i][j]}")
                count_mnx += np.array(list(action2action2mnx[i][j].values()))
            print("\n\n")
        
        print(f"Number of times per component in total mnx: {count_mnx}")


        
import numpy as np
import sys
from contextlib import closing
from io import StringIO
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class CliffWalkingEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.shape = (4, 5)
        self.start_state_index = np.ravel_multi_index((0, 0), self.shape)

        nS = np.prod(self.shape)
        nA = 4
        self.num_reward_components = 4
        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[2, 1:-1] = True
        self.empty_treasure = (0,self.shape[1]-1)
        self.filled_treasure = (2,self.shape[1]-1)
        self.gold = (2,2)
        self.monster = (1,self.shape[1]-1)

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(CliffWalkingEnv, self).__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        
        isDone = False
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        reward = np.zeros(self.num_reward_components)
        
        if self._cliff[tuple(new_position)]:
            reward[0] = -10
            isDone = True
        if self.empty_treasure == tuple(new_position):
            reward[3] = 1
            isDone = True
        if self.filled_treasure == tuple(new_position):
            reward[3] = 15
            isDone = True
        if self.gold == tuple(new_position):
            reward[1] = 10
        if self.monster == tuple(new_position):
            reward[2] = -20
            isDone = True
        
        return [(1.0, new_state, reward, isDone)]

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


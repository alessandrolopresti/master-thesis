import gym
from gym import spaces
import numpy as np
import random


class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # STATE VARIABLES
    # Type: Tuple((spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(5)))
    #   1. Face Expression Recognition (FER): Neutral (0), Happy (1), Sad (2)
    #   2. Speech Emotion Recognition (SER): Neutral (0), Happy (1), Sad (2)
    #   3. Object State: Ground (0), Close (1), Far (2)
    #   4. Environmental Sound: Television/Radio ON (0), People Talking (1), Environmental Noise (2), Sound Object (3), Falling Object (4)

    # ACTIONS
    # Type: Discrete(3)
    #   Num   Action
    #   0     Compliment: You're trustworthy. I'm so proud of you.
    #   1     Calming Action: Hey, don't worry, relax.
    #   2     Scolding: do not throw objects on the floor! It's not nice / Don't be silly!

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(5)))
        self.state = None
        self.done = False
        self.thresholdEndEpisode = 100  # it 100 iterations the episode ends. It is a hyperparameter, so you can change it to see what happens.
        self.visitedStates = 0  # it counts how many states the agent has visited so far, it is used because when this value is equals to the thresholdEndEpisode the episode ends.

        # We consider 5 regions in total to describe our state space
        # Notice that there will be some exceptions in the transition from one state to another taking some action

        self.visitableStates = {'NN0': [(0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3)],
                                'NN1': [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3)],
                                'NN2': [(0, 0, 0, 4)],
                                'HN0': [(0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3), (1, 0, 1, 0), (1, 0, 1, 1), (1, 0, 1, 2), (1, 0, 1, 3), (0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (1, 0, 2, 0), (1, 0, 2, 1), (1, 0, 2, 2), (1, 0, 2, 3)],
                                'HN1': [(0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2), (1, 0, 0, 3)],
                                'HN2': [(0, 1, 0, 4), (1, 0, 0, 4)],
                                'HH0': [(1, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 1, 3), (1, 1, 2, 0), (1, 1, 2, 1), (1, 1, 2, 2), (1, 1, 2, 3)],
                                'HH1': [(1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 0, 2), (1, 1, 0, 3)],
                                'HH2': [(1, 1, 0, 4)],
                                'SN0': [(0, 2, 1, 0), (0, 2, 1, 1), (0, 2, 1, 2), (0, 2, 1, 3), (2, 0, 1, 0), (2, 0, 1, 1), (2, 0, 1, 2), (2, 0, 1, 3), (0, 2, 2, 0), (0, 2, 2, 1), (0, 2, 2, 2), (0, 2, 2, 3), (2, 0, 2, 0), (2, 0, 2, 1), (2, 0, 2, 2), (2, 0, 2, 3)],
                                'SN1': [(0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 0, 2), (0, 2, 0, 3), (2, 0, 0, 0), (2, 0, 0, 1), (2, 0, 0, 2), (2, 0, 0, 3)],
                                'SN2': [(0, 2, 0, 4), (2, 0, 0, 4)],
                                'SS0': [(2, 2, 1, 0), (2, 2, 1, 1), (2, 2, 1, 2), (2, 2, 1, 3), (2, 2, 2, 0), (2, 2, 2, 1), (2, 2, 2, 2), (2, 2, 2, 3)],
                                'SS1': [(2, 2, 0, 0), (2, 2, 0, 1), (2, 2, 0, 2), (2, 2, 0, 3)],
                                'SS2': [(2, 2, 0, 4)]}

    # You pass one of the key in the dictionary and it returns one sample/state of that key
    def pickSampleByKey(self, key):
        indexSampleChosen = random.randrange(0, len(self.visitableStates[key]), 1)
        return self.visitableStates[key][indexSampleChosen]

    # Check if a sample belongs to a region
    def belongsTo(self, key):
        if (self.state in self.visitableStates[key]):
            return True
        return False

    # You pass one of the region and it returns one sample/state of that region
    # Since we consider 5 regions, we have: NN, HN, HH, SN , SS.
    # REMEMBER: We know that each region is characterize by 3 keys in the dictionary
    def pickSampleByRegion(self, region):
        temp = []
        for i in range(3):
            temp += self.visitableStates.get(region + str(i))
        indexSampleChosen = random.randrange(0, len(temp), 1)
        return temp[indexSampleChosen]

    def concatenateLists(self):
        temp = []
        for key in self.visitableStates:
            temp += self.visitableStates[key]
        return temp


    def step(self, action):
        if (np.random.random() <= 0.9):
            if (action == 0):
                if (self.belongsTo('NN0')):
                    self.state = self.pickSampleByRegion('HN')
                elif (self.belongsTo('NN1')):
                    self.state = self.pickSampleByRegion('SN')
                elif (self.belongsTo('NN2')):
                    self.state = self.pickSampleByRegion('SN')
                elif (self.belongsTo('HN0')):
                    self.state = self.pickSampleByRegion('HH')
                elif (self.belongsTo('HN1')):
                    self.state = self.pickSampleByRegion('NN')
                elif (self.belongsTo('HN2')):
                    self.state = self.pickSampleByRegion('NN')
                elif (self.belongsTo('HH0')):
                    self.state = self.pickSampleByRegion('HH')
                elif (self.belongsTo('HH1')):
                    self.state = self.pickSampleByRegion('HN')
                elif (self.belongsTo('HH2')):
                    self.state = self.pickSampleByRegion('HN')
                elif (self.belongsTo('SN0')):
                    self.state = self.pickSampleByRegion('SN')
                elif (self.belongsTo('SN1')):
                    self.state = self.pickSampleByRegion('SS')
                elif (self.belongsTo('SN2')):
                    self.state = self.pickSampleByRegion('SS')
                elif (self.belongsTo('SS0')):
                    self.state = self.pickSampleByRegion('SS')
                elif (self.belongsTo('SS1')):
                    self.state = self.pickSampleByRegion('SS')
                else: #(self.belongsTo('SS2')):
                    self.state = self.pickSampleByRegion('SS')
            elif (action == 1):
                if (self.belongsTo('NN0')):
                    self.state = self.pickSampleByRegion('NN')
                elif (self.belongsTo('NN1')):
                    self.state = self.pickSampleByRegion('NN')
                elif (self.belongsTo('NN2')):
                    self.state = self.pickSampleByRegion('SN')
                elif (self.belongsTo('HN0')):
                    self.state = self.pickSampleByRegion('HN')
                elif (self.belongsTo('HN1')):
                    self.state = self.pickSampleByRegion('HN')
                elif (self.belongsTo('HN2')):
                    self.state = self.pickSampleByRegion('NN')
                elif (self.belongsTo('HH0')):
                    self.state = self.pickSampleByRegion('HN')
                elif (self.belongsTo('HH1')):
                    self.state = self.pickSampleByRegion('HH')
                elif (self.belongsTo('HH2')):
                    self.state = self.pickSampleByRegion('HN')
                elif (self.belongsTo('SN0')):
                    self.state = self.pickSampleByRegion('NN')
                elif (self.belongsTo('SN1')):
                    self.state = self.pickSampleByRegion('SN')
                elif (self.belongsTo('SN2')):
                    self.state = self.pickSampleByRegion('SS')
                elif (self.belongsTo('SS0')):
                    self.state = self.pickSampleByRegion('SN')
                elif (self.belongsTo('SS1')):
                    self.state = self.pickSampleByRegion('SN')
                else: #(self.belongsTo('SS2')):
                    self.state = self.pickSampleByRegion('SS')
            else:
                if (self.belongsTo('NN0')):
                    self.state = self.pickSampleByRegion('SN')
                elif (self.belongsTo('NN1')):
                    self.state = self.pickSampleByRegion('HN')
                elif (self.belongsTo('NN2')):
                    self.state = self.pickSampleByRegion('HN')
                elif (self.belongsTo('HN0')):
                    self.state = self.pickSampleByRegion('NN')
                elif (self.belongsTo('HN1')):
                    self.state = self.pickSampleByRegion('HH')
                elif (self.belongsTo('HN2')):
                    self.state = self.pickSampleByRegion('HH')
                elif (self.belongsTo('HH0')):
                    self.state = self.pickSampleByRegion('HN')
                elif (self.belongsTo('HH1')):
                    self.state = self.pickSampleByRegion('HN')
                elif (self.belongsTo('HH2')):
                    self.state = self.pickSampleByRegion('HH')
                elif (self.belongsTo('SN0')):
                    self.state = self.pickSampleByRegion('SS')
                elif (self.belongsTo('SN1')):
                    self.state = self.pickSampleByRegion('NN')
                elif (self.belongsTo('SN2')):
                    self.state = self.pickSampleByRegion('NN')
                elif (self.belongsTo('SS0')):
                    self.state = self.pickSampleByRegion('SS')
                elif (self.belongsTo('SS1')):
                    self.state = self.pickSampleByRegion('SN')
                else: #(self.belongsTo('SS2')):
                    self.state = self.pickSampleByRegion('SN')
        else:
            self.state = self.concatenateLists()[random.randrange(0, len(self.visitableStates), 1)]

        reward = 0

        if (self.belongsTo('HH0') or self.belongsTo('HH1') or self.belongsTo('HH2')):
            reward += 1
        if (self.belongsTo('SS0') or self.belongsTo('SS1') or self.belongsTo('SS2')):
            reward -= 1

        self.visitedStates += 1
        if (self.visitedStates == self.thresholdEndEpisode):
            self.done = True

        return self.state, reward, self.done, "DEBUG"


    def reset(self):
        self.state = self.pickSampleByRegion('NN')
        self.done = False
        self.visitedStates = 0
        return self.state

    def render(self, mode='human', close=False):
        print("Render function called!")

    def getReward(self):
        return self.reward


    def getState(self):
        array = []
        for elem in self.state:
            array.append(elem)
        print("(", end='')
        if (array[0] == 0):
            print("Neutral", end='')
        if (array[0] == 1):
            print("Happy", end='')
        if (array[0] == 2):
            print("Sad", end='')
        print(", ", end='')
        if (array[1] == 0):
            print("Neutral", end='')
        if (array[1] == 1):
            print("Happy", end='')
        if (array[1] == 2):
            print("Sad", end='')
        print(", ", end='')
        if (array[2] == 0):
            print("Ground", end='')
        if (array[2] == 1):
            print("Close", end='')
        if (array[2] == 2):
            print("Far", end='')
        print(", ", end='')
        if (array[3] == 0):
            print("TV/Radio ON", end='')
        if (array[3] == 1):
            print("People talking", end='')
        if (array[3] == 2):
            print("Environmental Sound", end='')
        if (array[3] == 3):
            print("Sound Object", end='')
        if (array[3] == 4):
            print("Falling Object", end='')
        print(")")

    def getAction(self, action):
        print("Action: ", end='')
        if (action == 0):
            return "Compliment, you're such a good kid"
        elif (action == 1):
            return "Hey, do not worry! The therapy will end soon"
        else:
            return "Don't be silly and behave well otherwise the therapy will last longer"


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



    """"# It returns whether a sample is an exception or not
    def isException(self):
        if (self.state == (0, 0, 1, 0) or self.state == (0, 1, 1, 0) or self.state == (1, 0, 1, 0)
                or self.state == (1, 1, 1, 0) or self.state == (0, 2, 1, 0) or self.state == (2, 0, 1, 0)
                or self.state == (2, 2, 1, 0)):
            return True
        return False

    # It returns the values of the state variables corresponding to the indexes you specified in the array variables
    def takeValuesOfStateVariables(self, variables):
        array = []
        result = []
        for elem in self.state:
            array.append(elem)
        for i in range(len(variables)):
            result.append(array[variables[i]])
        return result
    """

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

    # This function is called to avoid that in a bad situation you have both emotionalFace and emotionalAudio equal to 'Neutral'
    """"" 
    def changeBadSituation(self, emotionalFace, emotionalAudio):
        if (np.random.random() <= 0.5):
            emotionalFace = 2
        else:
            emotionalAudio = 2
        return emotionalFace, emotionalAudio

    # This function is used to generate a situation that can be good or bad depending on the value of the flag parameter
    def generateSituation(self, situation, flag): # if flag == 1 then goodSituation else badSituation
        emotionalFace = situation['EmotionalFace'][0] if np.random.random() <= 0.2 else situation['EmotionalFace'][1]
        if(flag):
            fallingObject = situation['FallingObject'][0]
        else:
            fallingObject = situation['FallingObject'][0] if np.random.random() <= 0.2 else situation['FallingObject'][1]
        emotionalAudio = situation['EmotionalAudio'][0] if np.random.random() <= 0.1 else situation['EmotionalAudio'][1]
        usedObject = situation['UsedObject'][0] if np.random.random() <= 0.3 else situation['UsedObject'][1]

        # The last state variable has to follow a criterion:
        # completedTherapy can assume the value True when X good states are visited

        if (self.countGoodStates == self.thresholdEndEpisode):
            completedTherapy = 1
        else:
            completedTherapy = 0
        if (flag == 0 and emotionalFace == 0 and emotionalFace == emotionalAudio):
            emotionalFace, emotionalAudio = self.changeBadSituation(emotionalFace, emotionalAudio)
        return (emotionalFace, fallingObject, emotionalAudio, usedObject, completedTherapy)

    # This function returns if a situation is good or bad
    def kindOfSituation(self, situation):
        array = []
        for elem in situation:
            array.append(elem)
        for i in range(len(array)):
            if (array[0] == 1 or array[2] == 1): # array[0] == 1 means EmotionalFace is 'Happy', array[2] == 1 means EmotionalAudio is 'Happy'
                return 1 # return a good situation
            elif (array[0] == 0 and array[2] == 0): # array[0] == 0 means EmotionalFace is 'Neutral', array[2] == 1 means EmotionalAudio is 'Neutral'
                return 1
            else:
                return 0 # return a bad situation



    def howGood(self):
        print("Sono in HowGood")
        emotions = self.takeValuesOfStateVariables([0, 1, 2])
        if (emotions[0] == 0 and emotions[2] == 0): return 0
        if ((emotions[0] == 0 and emotions[2] == 1) or (emotions[0] == 1 and emotions[2] == 0)):    return 1
        if (emotions[0] == 1 and emotions[2] == 1): return 2

    def howBad(self):
        print("Sono in HowBad")
        emotions = self.takeValuesOfStateVariables([0, 1, 2])
        if ((emotions[0] == 0 and emotions[2] == 2) or (emotions[0] == 2 and emotions[2] == 0)): return -1
        if (emotions[0] == 2 and emotions[1] == 0 and emotions[2] == 2): return -2
        if (emotions[0] == 2 and emotions[1] == 1 and emotions[2] == 2): return -3

    def upState(self, level):
        if (level == -3): self.state = (2, 0, 2, 0, 0)
        if (level == -2): self.state = (0, 0, 2, 0, 0)
        if (level == -1): self.state = (0, 0, 0, 0, 0)
        if (level == 0): self.state = (1, 0, 0, 0, 0)
        if (level == 1): self.state = (1, 0, 1, 0, 0)
        if (level == 2): self.state = (1, 0, 1, 0, 0)
        return self.state

    def downState(self, level):
        if (level == 2): self.state = (1, 0, 0, 0, 0)
        if (level == 1): self.state = (0, 0, 0, 0, 0)
        if (level == 0): self.state = (0, 0, 2, 0, 0)
        if (level == -1): self.state = (2, 0, 2, 0, 0)
        if (level == -2): self.state = (2, 1, 2, 0, 0)
        if (level == -3): self.state = (2, 1, 2, 0, 0)
        return self.state


    def step(self, action):
        if (action == 0):
            if (self.takeValuesOfStateVariables([0, 2]) == [0, 0]):
                if (np.random.random() <= 0.5):





      # ACTION WAIT
        if action == 0:
            if (self.kindOfSituation(self.state)): # If I am in a good situation
                levelOfGoodness = self.howGood()
                if (np.random.random() <= 0.9): # with probability 90% the next state will be good
                    self.state = self.upState(levelOfGoodness)
                    if (self.howGood() == 2):
                        self.countGoodStates += 1
                else:
                    self.state = self.generateSituation(self.badSituation, 0)
            else:
                levelOfBadness = self.howBad()
                print(levelOfBadness)
                if (not levelOfBadness == -3):
                    if (np.random.random() <= 0.9):
                        self.state = self.downState(levelOfBadness)
                    else:
                        self.state = self.generateSituation(self.goodSituation, 1)
                        if (self.howGood() == 2):
                            self.countGoodStates += 1

        # ACTION HEY, DON'T WORRY.
        if action == 1:
            if (not self.kindOfSituation(self.state)): # If you are in bad situation
                levelOfBadness = self.howBad()
                self.state = self.upState(levelOfBadness)
            else: # Otherwise If you are in a good situation
                array = self.takeValuesOfStateVariables([0, 2])
                levelOfGoodness = self.howGood()
                if (array == [1, 1]):
                    self.state = self.downState(levelOfGoodness)

        # ACTION SOMETHING HAS FALLEN, I WILL TELL YOU A JOKE
        if (action == 2):
            if(self.kindOfSituation(self.state)): # if I am in a good state
                levelOfGoodness = self.howGood()
                self.state = self.downState(levelOfGoodness)
            else:
                levelOfBadness = self.howBad()
                if (self.takeValuesOfStateVariables([1]) == 1):
                    self.state = self.upState(levelOfBadness)
                else:
                    self.state = self.downState(levelOfBadness)

        self.reward -= 1
        valueCompletedTherapy = self.takeValuesOfStateVariables([4])
        if (valueCompletedTherapy[0] == 1):
            self.done = True

        return self.state, self.reward, self.done, "DEBUG" 
    """

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


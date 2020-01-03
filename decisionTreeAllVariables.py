import gym
import gym_foo
import numpy as np
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
from collections import deque
from sklearn.tree import DecisionTreeRegressor
import matplotlib.image as mpimg
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import scipy.stats


class decisionTreeAllVariables():

    def __init__(self, n_episodes=100, min_epsilon=0.01, gamma=0.9, ada_divisor=10):
        self.n_episodes = n_episodes # training episodes
        self.features = ["Face Expression Recognition", "Speech Emotion Recognition", "Object State", "Environmental Sound"]
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.ada_divisor = ada_divisor # only for development purposes
        self.env = gym.make('foo-v0')
        self.MAX_LEN = 1000 # this is the length of the buffer
        self.update = 600 # how many samples you generate from the new tree and update the buffer
        self.current_tree = DecisionTreeRegressor()
        self.buffer = deque(maxlen=self.MAX_LEN)
        self.X = deque(maxlen=self.MAX_LEN)
        self.y = deque(maxlen=self.MAX_LEN)
        self.slidingWindow = 5 # how many trees you want to consider
        self.num_run = 25 # how many times you want to run the algorithm from scratch
        self.num_trees_optimal_policy = 100 # number of trees you build after stopping the training and following the optimal policy (epsilon = 0)
        self.X_final = deque(maxlen=self.update * self.num_trees_optimal_policy) # buffer where we store samples of
        self.y_final = deque(maxlen=self.update * self.num_trees_optimal_policy)
        # meaning of indexes: 0 = Face Expression Recognition, 1 = Speech Emotion Recognition, 2 = Object State, 3 = Environmental Noise
        self.initialIndex = 0
        self.finalIndex = 4


    # Choosing action based on epsilon-greedy policy
    def choose_action(self, state, epsilon):
        # np.random.random() returns random floats in the half-open interval [0.0, 1.0).
        # The first time you fill the buffer, you must use epsilon = 1 (maximum exploration) because you do not have already a tree
        # Then you have q-values = 0, so the argmax will be zero for every q-value and then you will choose an action randomly
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.current_tree.predict([state])[0])


    def initializeBuffer(self):

        i = 0
        current_state = self.env.reset()
        total_reward = 0

        while(i < self.MAX_LEN):
            action = self.choose_action(current_state, 1) # maximum exploration, actions are chosen randomly
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            q = np.zeros(self.env.action_space.n)
            q[action] = reward
            current_state = current_state[self.initialIndex: self.finalIndex]
            self.buffer.append([current_state, action, obs, q]) # you are storing the reward in q, but in next steps you will store the Q-value of that action
            self.X.append(current_state)
            self.y.append(q)
            current_state = obs
            i += 1
            if (done):
                current_state = self.env.reset()

        print("Total reward of samples randomly chosen: " + str(total_reward))


    # Main algorithm to build decision trees
    def buildTree(self, demo=False):

        k = 0
        TEST_REWARD = 80.0

        # I wanto to know the average reward of all features after 25 run
        avgRewAllFeatures = []

        #  confidenceIntervalFeatures is a dictionary containing the Monte Carlo error considered for the self.num_trees_optimal policy trees for each run of the algorithm without considering the i-th feature (in this case the i-th key)
        confidenceIntervalFeatures = dict()
        for elem in self.features:
            confidenceIntervalFeatures[elem] = np.array([])

        # Average reward after self.n_runs for trees without a feature
        averageRewardFeatures = dict()
        for elem in self.features:
            averageRewardFeatures[elem] = np.array([])

        total_importance = np.ndarray(shape=(len(self.features), ), dtype=float) # it considers the importance of all runs of the algorithm

        while (k < self.num_run):
            print("RUN " + str(k))

            # first tree based on the randomly generated buffer
            self.initializeBuffer()

            # Fit regression model
            self.current_tree.fit(self.X, self.y)

            # i is used to count the number of updates a tree has done, j is used to update the value of epsilon
            # lastRun is used to count how many trees you want to build considering the optimal policy (epsilon = 0) after reaching the stopping criteria for training
            i = j = flag = lastRun = 0

            scores = []
            slidingWindowReward = []

            while (lastRun < self.num_trees_optimal_policy):

                print("Tree number: " + str(j))
                total_reward = 0
                epsilon = self.get_epsilon(j)
                current_state = self.env.reset()

                if (len(slidingWindowReward) == self.slidingWindow):
                    if (sum(slidingWindowReward)/len(slidingWindowReward) > TEST_REWARD):
                        total_importance += np.array(self.current_tree.feature_importances_)
                        print("Total importance: " + str(total_importance))
                        lastRun += 1
                        epsilon = 0 # in order to consider the optimal policy we set epsilon = 0
                    else:
                        slidingWindowReward = [] # we clear the slidingWindow if the samples considered don't match our threshold (TEST_REWARD)

                while (i < self.update):

                    action = self.choose_action(current_state, epsilon)
                    obs, reward, done, _ = self.env.step(action)
                    obs = obs[self.initialIndex:self.finalIndex]
                    total_reward += reward
                    q_current = self.current_tree.predict([current_state])
                    q_new = self.current_tree.predict([obs])
                    q_current[0][action] = reward + self.gamma * np.max(q_new[0])
                    self.buffer.append([current_state, action, obs, q_current[0]])
                    self.X.append(current_state)
                    self.y.append(q_current[0])

                    if (not lastRun == 0): # it means until we did not build self.num_trees_optimal_policy
                        self.X_final.append(current_state)
                        self.y_final.append(q_current[0])
                    current_state = obs
                    i += 1
                    if done:
                        current_state = self.env.reset()
                        if(not flag):
                            scores.append(total_reward)
                            flag = 1
                            if (lastRun == 0):
                                slidingWindowReward.append(self.test_reward())
                i = 0
                if (not flag):
                    scores.append(total_reward)
                    if (lastRun == 0):
                        slidingWindowReward.append(total_reward)

                self.current_tree = DecisionTreeRegressor()
                self.current_tree.fit(self.X, self.y)
                j += 1
                flag = 0


            # Testing considering all the variables but training the model on samples generated by last self.num_trees_optimal_policy with epsilon = 0
            print("Testing the tree considering all variables")
            self.current_tree = DecisionTreeRegressor()
            self.current_tree.fit(self.X_final, self.y_final)
            # I want to know the average reward after 25 run considering all variables
            avgRewAllFeatures.append(self.test_reward())

            if (not demo):
                # Visualize data
                if (self.MAX_LEN > 2 and self.update > 2):
                    dot_data = StringIO()
                    export_graphviz(self.current_tree, out_file=dot_data,
                                        filled=True, rounded=True,
                                        special_characters=True,
                                        feature_names=self.features)
                    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                    Image(graph.create_png())
                    graph.write_png("final_tree.png")


                # Now we build four trees (each one without considering one of the four features), we generate a test-set for each tree (so we generate some episodes)
                # From this episodes we compute the Monte-Carlo error that is (r + gamma * G_{t+1} - q^predicted(x,a))^{2} e we sum all these differences to compute the total error
                # Higher is this error, more important is the removed feature (so the feature we didn't consider)

                for i in range(0, len(self.features)):
                    print("Testing the final tree without considering " + str(self.features[i]))
                    self.current_tree = DecisionTreeRegressor()
                    X_feature = self.dataFilter(i)
                    self.current_tree.fit(X_feature, self.y_final)
                    r, e = self.getMonteCarloError(i)
                    averageRewardFeatures[self.features[i]] = np.append(averageRewardFeatures[self.features[i]], r)
                    confidenceIntervalFeatures[self.features[i]] = np.append(confidenceIntervalFeatures[self.features[i]], e)


            k += 1

        if (not demo):

            # This is the final average reward after 25 run considering all features
            print("FINAL AVERAGE REWARD CONSIDERING ALL FEATURES: ", end="")
            print(sum(avgRewAllFeatures)/len(avgRewAllFeatures))
            plt.plot(avgRewAllFeatures);
            plt.xlabel("Run")
            plt.ylabel("Reward")
            plt.title("Reward after " + str(self.num_run) + " run");
            plt.show()




            # This is the average total importance for all variables after 25 run
            print("TOTAL IMPORTANCE ACCORDING TO TREES: ", end="")
            print(total_importance/(self.num_run*self.num_trees_optimal_policy))



            x = ["Tree without FER", "Tree without SER", "Tree without OS", "Tree without ES"]
            y = []
            e = []
            for i in range(0, len(self.features)):
                m, inf, sup = self.mean_confidence_interval(confidenceIntervalFeatures[self.features[i]])
                print("Tree without considering feature: " + str(self.features[i]))
                print("Average reward: " + str(sum(averageRewardFeatures[self.features[i]])/len(averageRewardFeatures[self.features[i]])))
                print("Inferior interval: " + str(inf) + " Average: " + str(m) + " Superior Interval: " + str(sup))
                y.append(m)
                e.append(sup-m)

            plt.errorbar(x, y, yerr=e, fmt='o')
            plt.xlabel("Trees considering three features")
            plt.ylabel("Monte Carlo Error")
            plt.show()





    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m - h, m + h


    def getMonteCarloError(self, index):
        i = 0
        current_state = self.env.reset()
        total_reward = 0
        num_episodes = 100
        buffer = deque()
        while (i < num_episodes):
            listx = list(current_state)
            listx.remove(current_state[index])
            tuplex = tuple(listx)
            current_state = tuplex
            action = self.choose_action(current_state, 0)  # maximum exploration, actions are chosen randomly
            obs, reward, done, _ = self.env.step(action)
            buffer.append([reward, self.current_tree.predict([current_state])[0][action]])
            total_reward += reward
            current_state = obs
            if (done):
                current_state = self.env.reset()
                i += 1

        avg_r = total_reward/num_episodes
        #print("Average reward for " + str(num_episodes) + " episodes: " + str(avg_r))

        j = len(buffer) - 1
        num = self.n_episodes - 1
        error = []
        g_values = []
        while(j >= 0):
            if (num == self.n_episodes - 1):
                g_t = buffer[j][0]
            else:
                g_t = buffer[j][0] + (self.gamma * g_values[len(g_values)-1])

            g_values.append(g_t)
            error_t = abs(g_t - buffer[j][1])
            error.append(error_t)
            num -= 1
            num -= 1
            j -= 1
            if (num == -1):
                num = self.n_episodes - 1

        final_error = sum(error)/num_episodes
        return avg_r, final_error


    # Adaptive learning of Exploration Rate
    def get_epsilon(self, t):
        # t corresponds to the n-th episode, so it is an integer
        # According to the formula below, the value of both epsilon and alpha are 1.0
        # So their values decrease according to the formula below until the min_epsilon and min_alpha values
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    # Adaptive learning of Learning Rate
    def get_alpha(self, t):
        return max(self.min_alpha, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def clearDeque(self):
        self.buffer.clear()
        self.X.clear()
        self.y.clear()

    def plot_scores(self, scores, flag):
        """Plot scores and optional rolling mean using specified window."""
        plt.figure(flag)
        plt.plot(scores);
        if (flag == 1):
            plt.title("Training Scores");
        else:
            plt.title("Test Scores")

        mylist =scores
        N = 10
        cumsum, moving_aves = [0], []

        for i, x in enumerate(mylist, 1):
            cumsum.append(cumsum[i - 1] + x)
            if i >= N:
                moving_ave = (cumsum[i] - cumsum[i - N]) / N
                # can do stuff with moving_ave here
                moving_aves.append(moving_ave)

        rolling_mean = pd.Series(scores).rolling(self.n_episodes).mean()
        print (moving_aves)
        plt.plot(moving_aves);
        plt.show()

        return rolling_mean

    def test_reward(self):
        scores = []
        for e in range(200):
            done = False
            total_reward = 0
            epsilon = 0
            i = 0
            current_state = self.env.reset()

            while not done:
                # Render environment
                # self.env.render()
                # Choose action according to greedy policy and take it
                current_state = current_state[self.initialIndex:self.finalIndex]
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                current_state = obs
                i += 1

            if done:
                if e % 100 == 0:
                    print("[EPISODE " + str(e) + " running]")
            scores.append(total_reward)

        #_ = self.plot_scores(scores, 0)
        #plt.show()
        print("Average reward: " + str(sum(scores) / len(scores)))
        self.clearDeque()
        return sum(scores)/len(scores)


    def test_model(self):
        scores = []
        for e in range(200):
            done = False
            total_reward = 0
            epsilon = 0
            i = 0
            current_state = self.env.reset()

            while not done:
                # Render environment
                #self.env.render()
                # Choose action according to greedy policy and take it
                current_state = current_state[self.initialIndex:self.finalIndex]
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                current_state = obs
                i += 1

            if done:
                if e % 100 == 0:
                    print("[EPISODE " + str(e) + " running]")
            scores.append(total_reward)

        #_ = self.plot_scores(scores, 0)
        #plt.show()
        print("Average reward: " + str(sum(scores)/len(scores)))
        self.env.close()
        self.clearDeque()


    def dataFilter(self, index):
        X_temp = deque(maxlen=self.update * self.num_trees_optimal_policy) # It's the last number of tree you train without exploration, epsilon = 0
        for elem in self.X_final:
            listx = list(elem)
            listx.remove(elem[index])
            tuplex = tuple(listx)
            X_temp.append(tuplex)
        return X_temp


if __name__ == "__main__":

    # Make an instance of CartPole class
    solver = decisionTreeAllVariables()
    solver.buildTree(False)

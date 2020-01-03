import gym
import gym_foo
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd



class Tiago():
    def __init__(self, n_episodes=100, min_alpha=0.1, min_epsilon=0.1, gamma=0.9, ada_divisor=10, max_env_steps=None, monitor=False):
        self.n_episodes = n_episodes # training episodes
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.ada_divisor = ada_divisor # only for development purposes

        self.env = gym.make('foo-v0')
        env_struct = (3, 3, 3, 5,)
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        if monitor: self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1', force=True) # record results for upload

        # initialising Q-table
        self.Q = np.zeros(env_struct + (self.env.action_space.n,)) # so the Q-table shape is (1, 1, 6, 12, 2)

    # Choosing action based on epsilon-greedy policy
    def choose_action(self, state, epsilon):
        # np.random.random() returns random floats in the half-open interval [0.0, 1.0).
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    # Updating Q-value of state-action pair based on the update equation
    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    # Adaptive learning of Exploration Rate
    def get_epsilon(self, t):
        # t corresponds to the n-th episode, so it is an integer
        # According to the formula below, the value of both epsilon and alpha are 1.0
        # So their values decrease according to the formula below until the min_epsilon and min_alpha values
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    # Adaptive learning of Learning Rate
    def get_alpha(self, t):
        return max(self.min_alpha, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def run(self):

        scores = []

        for e in range(self.n_episodes):
            # As states are continuous, discretize them into buckets
            current_state = self.env.reset()
            #self.env.getState()
            # Get adaptive learning alpha and epsilon decayed over time
            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            total_reward = 0
            i = 0

            while not done:
                # Render environment
                #self.env.render()

                # Choose action according to greedy policy and take it
                action = self.choose_action(current_state, epsilon)
                #self.env.getAction(action)
                obs, reward, done, _ = self.env.step(action)
                self.env.getState()
                total_reward += reward
                new_state = obs

                # Update Q-Table
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1


            if done and e % 10 == 0:
                print(total_reward)
                print("[EPISODE " + str(e) + " running]")

            scores.append(total_reward)

        _ = self.plot_scores(scores, 1)
        plt.show()
        print(self.Q)

    def plot_scores(self, scores, flag):
        """Plot scores and optional rolling mean using specified window."""
        plt.figure(flag)
        plt.plot(scores);
        if (flag == 1):
            plt.title("Training Scores");
        else:
            plt.title("Test Scores")

        mylist =scores
        N = 200
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

    def test_model_2(self):
        done = False
        total_reward = 0
        epsilon = 0
        i = 0
        current_state = self.env.reset()
        self.env.getState()
        while not done:
            action = self.choose_action(current_state, epsilon)
            self.env.getAction(action)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            current_state = obs
            i += 1
        print("Total reward: " + str(total_reward))





    def test_model(self):
        scores = []
        print(self.Q)
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
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                current_state = obs
                i += 1

            if done:
                print(total_reward)
                if e % 100 == 0:
                    print("[EPISODE " + str(e) + " running]")
            scores.append(total_reward)

        _ = self.plot_scores(scores, 0)
        plt.show()
        self.env.close()



if __name__ == "__main__":
    model = Tiago()
    model.run()
    model.test_model()
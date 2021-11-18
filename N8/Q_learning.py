import numpy as np
import pandas as pd
import Env_8 as Env
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.99, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def choose_action_greedy(self, observation):
        self.check_state_exist(observation)
        state_action = self.q_table.loc[observation, :]
        # some actions may have the same value, randomly choose on in these actions
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

if __name__ == "__main__":
    env = Env.Environ()
    RL = QLearningTable(actions=list(range(2)))

    xar = []
    yar = []
    training_time = 0
    for i in tqdm(range(50)):
        score_avg = 0
        for j in range(10):
            state = env.reset()
            trajectory = []
            while True:
                action = RL.choose_action(str(state))
                next_state, reward, done, action, cost = env.step(action, trajectory)
                time_start = time.time()
                RL.learn(str(state), action, reward, str(next_state))
                training_time += (time.time() - time_start)
                state = next_state
                trajectory.append((next_state, action, reward, cost))
                if done:
                    score_avg += reward
                    break
        xar.append(i)
        yar.append(score_avg/10)
    print(training_time)
    plt.plot(xar, yar, linewidth=3)
    plt.title("Q_learning", fontsize=19)
    plt.xlabel("Episodes", fontsize=10)
    plt.ylabel("Reward", fontsize=10)
    plt.tick_params(axis='both', labelsize=9)
    #plt.show()
    np.savetxt("data/Q_learning_N_8.txt", yar)

    len_record = []
    cost_record = []
    reward_record = []
    #
    # start = time.time()
    #
    # for j in range(100):
    #     state = env.reset()
    #     trajectory = []
    #     for i in range(100):
    #         optimal_a = RL.choose_action_greedy(str(state))
    #         next_state, reward, done, action, cost = env.step(action, trajectory)
    #         state = next_state
    #         trajectory.append((next_state, action, reward, cost))
    #         if done:
    #             len_record.append(len(trajectory))
    #             cost_record.append(1000 - reward)
    #             reward_record.append(reward)
    #             break
    #
    # print('running time every step:')
    # print((time.time() - start) / 100)
    #
    # len_record = np.array(len_record)
    # cost_record = np.array(cost_record)
    # print('mean, std, min, max of timeslot')
    # print(np.mean(len_record), np.std(len_record), np.min(len_record), np.max(len_record))
    # print('mean, std, min, max of reward')
    # print(np.mean(reward_record), np.std(reward_record), np.min(reward_record), np.max(reward_record))



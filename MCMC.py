import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import env as Env
env = Env.Environ()
env_train = Env.Environ()

def make_epsilon_greedy_policy(Q_table, nA, epsilon):
    def generate_policy(observation):
        prob_A = np.ones(nA)*epsilon / nA
        optimal_a = np.argmax(Q_table[observation])
        prob_A[optimal_a] += (1.0 - epsilon)
        return prob_A
    return generate_policy

def generate_one_episode(env, generate_policy):
    trajectory = []
    state = env.reset()
    while True:
        Pi_table = generate_policy(state)
        action = np.random.choice(np.arange(len(Pi_table)), p=Pi_table)
        next_state, reward, done, action1, cost = env.step(action, trajectory)
        trajectory.append((state, action1, reward, cost))
        if done:
            break
        state = next_state
    return trajectory

def MC_control(env, epoch=1, episode_len=100, epsilon=0.5, discount_factor=0.99):
    Return, Count = defaultdict(float), defaultdict(float)
    policy = make_epsilon_greedy_policy(Q_table, env.action_space_n, epsilon)
    for k in range(epoch):
        score_avg = 0
        for j in range(episode_len):
            trajectory = generate_one_episode(env, policy)
            for x in trajectory:
                if x[2] > 0:
                    score_avg += x[2]
            if len(trajectory) <= env.deadline:
                s_a_pairs = set([(x[0], x[1]) for x in trajectory])
                for state, action in s_a_pairs:
                    s_a = (state, action)
                    first_visit_id = next(i for i, x in enumerate(trajectory) if x[0] == state and x[1] == action)
                    G = sum([x[2]*(discount_factor**i) for i, x in enumerate(trajectory[first_visit_id:])])
                    Return[s_a] += G
                    Count[s_a] += 1.
                    Q_table[state][action] = Return[s_a] / Count[s_a]
    return policy


Q_table = defaultdict(lambda: np.zeros(env.action_space_n))
#optimal_policy = MC_control(env_train)
# 用户
cost_record = []
k = 0
ans = 0
for i in range(500):
    # print("第"+str(i+1)+"个用户：")
    print("第"+str(i+1)+"次用户应用的运行过程：")
    state_init = env.reset()
    trajectory1 = []
    changdu = 0
    while True:
        changdu += 1
        s1 = time.time()
        # if len(Q_table) < 3000:
        #     optimal_policy = MC_control(env_train)
        #     print("第" + str(k + 1) + "次训练时间(单位s)(500次)：", time.time() - s1)
        #     # print(len(Q_table))
        optimal_policy = MC_control(env_train)
        # print("第" + str(k + 1) + "次训练时间(单位s)(100次)：", time.time() - s1)
        k += 1
        optimal_a = np.argmax(Q_table.get((state_init[0], state_init[1])))
        # print(Q_table.get((state_init[0], state_init[1])))
        # print(optimal_a)
        #print("选择动作是:(0不迁移1迁移):", optimal_a)
        next_state, reward, done, action, cost = env.step(optimal_a, trajectory1)
        state_init = next_state
        trajectory1.append((next_state, action, reward, cost))
        if done:
            cost_record.append(1000 - reward)
            print("第" + str(i + 1) + "次用户的cost：", 1000 - reward)
            print()
            print()
            break
    if changdu <= env.deadline:
        ans += 1

cost_record = np.array(cost_record)
print("所有次用户的平均cost：",np.mean(cost_record))
print("传输成功率：",ans/500)
# plt.plot(range(len(cost_record)), cost_record, linewidth=3)
# plt.title("MCMC", fontsize=19)
# plt.xlabel("Iteration", fontsize=10)
# plt.ylabel("Cost", fontsize=10)
# plt.tick_params(axis='both', labelsize=9)
# plt.show()








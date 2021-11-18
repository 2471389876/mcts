import numpy as np
import env as Env
env = Env.Environment()

len_record = []
cost_record = []
reward_record = []

deadline_count = 0
for j in range(1000):
    state = env.reset()
    trajectory = []
    for i in range(100):
        optimal_a = 1
        next_state, reward, done, action, cost = env.step(optimal_a, trajectory)
        #print(state, action, next_state, reward, cost)
        #print(state)
        state = next_state
        trajectory.append((next_state, action, reward, cost))

        if done:
            len_record.append(len(trajectory))
            cost_record.append(1000-reward)
            reward_record.append(reward)
            #print(1000-reward)

            if len(trajectory) <= env.Deadline:
                deadline_count += 1
            break

print('服务成功率：')
print(deadline_count/1000)

len_record = np.array(len_record)
cost_record = np.array(cost_record)
reward_record = np.array(reward_record)
print('一直迁移的cost：')
print(np.mean(cost_record))





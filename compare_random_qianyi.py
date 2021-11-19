import random
import numpy as np
import Env_8 as Env
import datetime
from common import *
env = Env.Environ()

len_record = []
cost_record = []
reward_record = []

deadline_count = 0
for j in range(1000):
    #print("第"+str(j+1)+"次：")
    state = env.reset()
    trajectory = []
    while(True):

        #random.seed(datetime.datetime.now())
        if random.random() <0.3 :
            optimal_a = 1
        else:
            optimal_a = 0
        #print(optimal_a)
        next_state, reward, done, action, cost = env.step(optimal_a, trajectory)
        #print(state, action, next_state, reward, cost)
        #print(state,action)
        state = next_state
        trajectory.append((next_state, action, reward, cost))

        if done:
            len_record.append(len(trajectory))
            cost_record.append(1000-reward)
            reward_record.append(reward)
            #print(1000-reward)
            if len(trajectory) <= env.deadline:
                deadline_count += 1
            break


print('服务成功率：')
print(deadline_count/1000)
logging.info(f"random transfer strategy's service success ratio:{deadline_count/1000}")

len_record = np.array(len_record)
cost_record = np.array(cost_record)
reward_record = np.array(reward_record)
print('随机迁移的cost：')
print(np.mean(cost_record))
logging.info(f"random transfer strategy's cost:{np.mean(cost_record)}")

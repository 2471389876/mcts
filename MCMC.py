import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import env as Env
envForRun = Env.Environment()
envForTrain = Env.Environment()

def MakeEpsilonGreedyPolicy(qTable, numAction, epsilon):
    def GeneratePolicy(observation):
        probAction = np.ones(numAction)*epsilon / numAction
        optimalAction = np.argmax(qTable[observation])
        probAction[optimalAction] += (1.0 - epsilon)
        return probAction
    return GeneratePolicy

def GenerateOneEpisode(env, generatePolicy):
    trajectory = []
    state = env.reset()
    while True:
        piTable = generatePolicy(state)
        action = np.random.choice(np.arange(len(piTable)), p=piTable)
        nextState, reward, done, action, cost = env.step(action, trajectory)
        trajectory.append((state, action, reward, cost))
        if done:
            break
        state = nextState
    return trajectory

def McmcControl(env, epoch=1, episodeLen=100, epsilon=0.5, discountFactor=0.99):
    Return, Count = defaultdict(float), defaultdict(float)
    policy = MakeEpsilonGreedyPolicy(qTable, env.ActionDim, epsilon)
    for k in range(epoch):
        scoreAvg = 0
        for j in range(episodeLen):
            trajectory = GenerateOneEpisode(env, policy)
            for x in trajectory:
                if x[2] > 0:
                    scoreAvg += x[2]
            if len(trajectory) <= env.Deadline:
                stateActionPairs = set([(x[0], x[1]) for x in trajectory])
                for state, action in stateActionPairs:
                    stateAndAction = (state, action)
                    firstVisitId = next(i for i, x in enumerate(trajectory) if x[0] == state and x[1] == action)
                    G = sum([x[2]*(discountFactor**i) for i, x in enumerate(trajectory[firstVisitId:])])
                    Return[stateAndAction] += G
                    Count[stateAndAction] += 1.
                    qTable[state][action] = Return[stateAndAction] / Count[stateAndAction]
    return policy


qTable = defaultdict(lambda: np.zeros(envForRun.ActionDim))
# 用户
costRecordList = []
trainingCount = 0
successTransferCount = 0
for i in range(10):
    print("第"+str(i+1)+"次用户应用的运行过程：")
    state = envForRun.reset()
    trajectory = []
    stepCount = 0
    while True:
        stepCount += 1
        optimalPolicy = McmcControl(envForTrain)
        trainingCount += 1
        optimalAction = np.argmax(qTable.get((state[0], state[1])))
        print(optimalAction)
        nextState, reward, done, action, cost = envForRun.step(optimalAction, trajectory)
        print(action)
        state = nextState
        trajectory.append((nextState, action, reward, cost))
        # print(trajectory)
        if done:
            costRecordList.append(1000 - reward)
            print("第" + str(i + 1) + "次用户的cost：", 1000 - reward)
            print()
            print()
            break
    if stepCount <= envForRun.Deadline:
        successTransferCount += 1

costRecordList = np.array(costRecordList)
print("所有次用户的平均cost：",np.mean(costRecordList))
print("传输成功率：",successTransferCount/10)
# plt.plot(range(len(cost_record)), cost_record, linewidth=3)
# plt.title("MCMC", fontsize=19)
# plt.xlabel("Iteration", fontsize=10)
# plt.ylabel("Cost", fontsize=10)
# plt.tick_params(axis='both', labelsize=9)
# plt.show()








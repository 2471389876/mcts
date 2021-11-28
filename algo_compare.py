import logging
import numpy as np
import env as Env
import argparse
import random
from common import *

parser = argparse.ArgumentParser()
# 算法有never, heuristic, always, random
parser.add_argument('--Algo', default="never", type=str) 
parser.add_argument('--TotalEpisodesNum', default=1000, type=int)
env = Env.Environment()

def chooseStragey(args, dis):
    if args.Algo == "never":
        return 0
    elif args.Algo == "always":
        return 1
    elif args.Algo == "random":
        if random.random() <0.3 :
            return 1
        else:
            return 0
    elif args.Algo == "heuristic":
        if dis > 3:
            return 1
        else:
            return 0


def main(args):
    lenRecord = []
    costRecord = []
    rewardRecord = []

    successTransferCount = 0
    for j in range(args.TotalEpisodesNum):
        state = env.reset()
        trajectory = []
        while True:
            optimal_a = chooseStragey(args, env.agent.State[0])
            next_state, reward, done, action, cost = env.step(optimal_a, trajectory)
            state = next_state
            trajectory.append((next_state, action, reward, cost))

            if done:
                lenRecord.append(len(trajectory))
                costRecord.append(1000-reward)
                rewardRecord.append(reward)

                if len(trajectory) <= env.Deadline:
                    successTransferCount += 1
                break
    costRecord = np.array(costRecord)
    costAvg = np.mean(costRecord)
    logging.info(f"{args.Algo} transfer strategy's service success ratio:{successTransferCount/args.TotalEpisodesNum}")
    logging.info(f"{args.Algo} transfer strategy's cost:{np.mean(costRecord)}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)





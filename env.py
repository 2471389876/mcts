#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import random
from agent import Agent

class Environment:
    def __init__(self):
        self.ActionDim = 2
        self.MaxDistance = 6  # 最大距离？
        self.Deadline = 9  # 最大步数
        self.TimeTotal = 5 # 一个决策时隙有5s
        self.TimeForTraining = 0.06 # 训练时间0.05s
        self.NoTranCostUnit= 2 # 不迁移成本（单位/s） 
        self.TranCostUnit = 50 # 迁移成本 
        self.RateInDiffDistanceList = [49.94, 29.844, 29.484, 28.884, 28.044, 26.964, 25.644]
        self.TransferInDiffDistanceList = [0, 30, 29.88, 29.52, 28.92, 27.0, 25.68]
        self.DataSize = 10 # 两个微云之间迁移的相关数据,设为10M
        self.P1 = 0.2
        self.P2 = 0.2
        self.P3 = 0.2
        self.P4 = 0.2

        self.agent = Agent()

    def reset(self):
        self.agent.reset()
        return (self.agent.State[0], self.agent.State[1], self.agent.State[2])

    def calTransferTime(self, curDis):
        return 0.02 * curDis * curDis

    def randChoiceTwoSelections(self, point, according, selections):
        if point <= according:
            return selections[0]
        else:
            return selections[1]

    def randChoiceThreeSelections(self, point, according1, according2, selections):
        if point <= according1:
            return selections[0]
        elif point > according2:
            return selections[2]
        else:
            return selections[1]

    def disChangeNoTrans(self):
        curDis = self.agent.State[0]
        if curDis == 0:
            return self.randChoiceTwoSelections(random.random(), self.P1, [curDis, curDis+1])
        elif curDis == self.MaxDistance:
            return self.randChoiceTwoSelections(random.random(), self.P4, [curDis-1, curDis])
        else:
            return self.randChoiceThreeSelections(random.random(), self.P2, self.P3, [curDis-1, curDis, curDis+1])
    
    def disChangeTrans(self):
        return self.randChoiceTwoSelections(random.random(), self.P1, [0, 1])


    def step(self, action, trajectory):
        done = 0
        cost = 0
        reward = 0

        if self.agent.State[0] == 0: # 距离为0时不存在迁移(a=1)的动作
            action = 0
        if action == 0:
            curDis = self.agent.State[0]
            self.agent.State[1] -= self.RateInDiffDistanceList[curDis] * self.TimeTotal
            cost = self.NoTranCostUnit * self.TimeTotal
            
            # 数据传输完成,轨迹结束
            if self.agent.State[1] < 0:
                self.agent.State[1] = 0

                totalCost = 0
                for i in range(len(trajectory)):
                    totalCost += trajectory[i][3]
                totalCost += cost
                reward = 1000 - totalCost
                done = 1

            self.agent.State[0] = self.disChangeNoTrans()
        else: 
            curDis = self.agent.State[0]
            transferTime = self.DataSize / self.TransferInDiffDistanceList[1] + self.calTransferTime(curDis)
            oldDisTransferTime = self.TimeForTraining
            newDisTransferTime = self.TimeTotal - transferTime - oldDisTransferTime
            self.agent.State[1] -= self.RateInDiffDistanceList[curDis] * oldDisTransferTime
            self.agent.State[1] -= self.RateInDiffDistanceList[0] * newDisTransferTime
            cost = self.TranCostUnit + self.NoTranCostUnit * self.TimeTotal

            if self.agent.State[1] <= 0:
                self.agent.State[1] = 0

                totalCost = 0
                for i in range(len(trajectory)):
                    totalCost += trajectory[i][3]
                totalCost += cost
                reward = 1000 - totalCost
                done = 1

            self.agent.State[0] = self.disChangeTrans()
            
        self.agent.State[2] += 1

        return ((self.agent.State[0], self.agent.State[1]), reward, done, action, cost)
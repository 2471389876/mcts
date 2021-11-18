import numpy as np
import random

class Environ:

    def __init__(self):
        self.maxdistance = 6
        self.deadline = 9
        self.action_space_n = 2
        self.state = np.zeros(3)
        self.timeslot = 5 
        self.t_new = 0.05#xunlian_time
        self.uc1 = 2
        self.uc2 = 50 # 迁移
        # self.rate = [5, 4.68, 3.61, 3.55, 2.03, 1.36, 0.59, 0.21]
        # self.transfer = [0, 100, 50, 40, 37.5, 35, 32.5, 30]
        self.rate = [49.94, 29.844, 29.484, 28.884, 28.044,26.964,25.644]
        #self.rate = [48, 29, 29, 29, 29, 28, 25, 25]
        self.transfer = [0, 30, 29.88, 29.52, 28.92,27,25.68]
        # self.M = 100
        self.M = 10
        self.Punish = 0
        self.P1 = 0.2
        self.P2 = 0.2
        self.P3 = 0.2
        self.P4 = 0.2

    def reset(self):
        # self.state[0] = random.randint(0,6)
        self.state[0] = 1
        self.state[1] = 2000
        self.state[2] = 0

        return (self.state[0], self.state[1],self.state[2])

    def step(self, action, trajectory):  # next_state, reward, done, action
        done = 0
        reward = 0
        cost = 0
        self.state[2] += 1
        if self.state[0] == 0:  # 不迁移
            if action == 1:
                action = 0
            cost = self.uc1 * self.timeslot
            self.state[1] -= self.rate[0] * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                #reward = 1000 - cost_last
                done = 1
            judge = random.random()
            if judge < self.P1:
                self.state[0] = 0
            else:
                self.state[0] = 1

        elif self.state[0] == 1 and action == 0:
            cost = self.uc1 * self.timeslot
            self.state[1] -= self.rate[1] * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            if judge < self.P2:
                self.state[0] = 0
            elif judge > self.P3:
                self.state[0] = 2

        elif self.state[0] == 1 and action == 1:
            t_stop = self.M / self.transfer[1] + 0.02
            t_2 = self.timeslot - t_stop - self.t_new
            self.state[1] -= self.rate[1] * self.t_new
            self.state[1] -= self.rate[0] * t_2
            # cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
            cost = self.uc2  + self.uc1 * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            if judge < self.P1:
                self.state[0] = 0
            else:
                self.state[0] = 1

        elif self.state[0] == 2 and action == 0:
            cost = self.uc1 * self.timeslot
            self.state[1] -= self.rate[2] * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            if judge < self.P2:
                self.state[0] = 1
            elif judge > self.P3:
                self.state[0] = 3

        elif self.state[0] == 2 and action == 1:
            t_stop = self.M / self.transfer[1]+0.08
            t_2 = self.timeslot - t_stop - self.t_new
            self.state[1] -= self.rate[2] * self.t_new
            self.state[1] -= self.rate[0] * t_2
            #cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
            cost = self.uc2 + self.uc1 * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            if judge < self.P1:
                self.state[0] = 0
            else:
                self.state[0] = 1

        elif self.state[0] == 3 and action == 0:
            cost = self.uc1 * self.timeslot
            self.state[1] -= self.rate[3] * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            if judge < self.P2:
                self.state[0] = 2
            elif judge > self.P3:
                self.state[0] = 4

        elif self.state[0] == 3 and action == 1:
            t_stop = self.M / self.transfer[1] + 0.18
            t_2 = self.timeslot - t_stop - self.t_new
            self.state[1] -= self.rate[3] * self.t_new
            self.state[1] -= self.rate[0] * t_2
           #cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
            cost = self.uc2 + self.uc1 * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            if judge < self.P1:
                self.state[0] = 0
            else:
                self.state[0] = 1

        elif self.state[0] == 4 and action == 0:
            cost = self.uc1 * self.timeslot
            self.state[1] -= self.rate[4] * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            # if judge < self.P4:
            #     self.state[0] = 3
            if judge < self.P2:
                self.state[0] = 3
            elif judge > self.P3:
                self.state[0] = 5

        elif self.state[0] == 4 and action == 1:
            t_stop = self.M / self.transfer[1] + 0.32
            t_2 = self.timeslot - t_stop - self.t_new
            self.state[1] -= self.rate[4] * self.t_new
            self.state[1] -= self.rate[0] * t_2
            #cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
            cost = self.uc2 + self.uc1 * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            if judge < self.P1:
                self.state[0] = 0
            else:
                self.state[0] = 1

        elif self.state[0] == 5 and action == 0:
            cost = self.uc1 * self.timeslot
            self.state[1] -= self.rate[5] * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            # if judge < self.P4:
            #     self.state[0] = 4
            if judge < self.P2:
                self.state[0] = 4
            elif judge > self.P3:
                self.state[0] = 6

        elif self.state[0] == 5 and action == 1:
            t_stop = self.M / self.transfer[1] + 0.5
            t_2 = self.timeslot - t_stop - self.t_new
            self.state[1] -= self.rate[5] * self.t_new
            self.state[1] -= self.rate[0] * t_2
            #cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
            cost = self.uc2 + self.uc1 * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            if judge < self.P1:
                self.state[0] = 0
            else:
                self.state[0] = 1

        elif self.state[0] == 6 and action == 0:
            cost = self.uc1 * self.timeslot
            self.state[1] -= self.rate[6] * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            if judge < self.P4:
                self.state[0] = 5
            # if judge < self.P2:
            #     self.state[0] = 5
            # elif judge > self.P3:
            #     self.state[0] = 7

        elif self.state[0] == 6 and action == 1:
            t_stop = self.M / self.transfer[1] + 0.72
            t_2 = self.timeslot - t_stop - self.t_new
            self.state[1] -= self.rate[6] * self.t_new
            self.state[1] -= self.rate[0] * t_2
            #cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
            cost = self.uc2 + self.uc1 * self.timeslot
            if self.state[1] <= 0:
                self.state[1] = 0
                cost_last = 0
                for i in range(len(trajectory)):
                    cost_last += trajectory[i][3]
                cost_last += cost
                if len(trajectory) >= self.deadline:
                    reward = 1000 - cost_last
                else:
                    reward = 1000 - cost_last
                # reward = 1000 - cost_last
                done = 1
            judge = random.random()
            if judge < self.P1:
                self.state[0] = 0
            else:
                self.state[0] = 1

        # elif self.state[0] == 7 and action == 0:
        #     cost = self.uc1 * self.timeslot
        #     self.state[1] -= self.rate[7] * self.timeslot
        #     if self.state[1] <= 0:
        #         self.state[1] = 0
        #         cost_last = 0
        #         for i in range(len(trajectory)):
        #             cost_last += trajectory[i][3]
        #         cost_last += cost
        #         if len(trajectory) >= self.deadline:
        #             reward = 1000 - cost_last
        #         else:
        #             reward = 1000 - cost_last
        #         # reward = 1000 - cost_last
        #         done = 1
        #     judge = random.random()
            # if judge < self.P2:
            #     self.state[0] = 6
            # elif judge>self.P3:
            #     self.state[0] = 8
            # if judge < self.P4:
            #     self.state[0] = 6

        # elif self.state[0] == 7 and action == 1:
        #     t_stop = self.M / self.transfer[1] + 0.98
            # t_2 = self.timeslot - t_stop - self.t_new
            # self.state[1] -= self.rate[7] * self.t_new
            # self.state[1] -= self.rate[0] * t_2
            # #cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
            # cost = self.uc2 + self.uc1 * self.timeslot
            # if self.state[1] <= 0:
            #     self.state[1] = 0
            #     cost_last = 0
            #     for i in range(len(trajectory)):
            #         cost_last += trajectory[i][3]
            #     cost_last += cost
            #     if len(trajectory) >= self.deadline:
            #         reward = 1000 - cost_last
            #     else:
            #         reward = 1000 - cost_last
                # reward = 1000 - cost_last
            #     done = 1
            # judge = random.random()
            # if judge < self.P1:
            #     self.state[0] = 0
            # else:
            #     self.state[0] = 1
        
        # elif self.state[0] == 8 and action == 0:
        #     cost = self.uc1 * self.timeslot
        #     self.state[1] -= self.rate[8] * self.timeslot
            # if self.state[1] <= 0:
            #     self.state[1] = 0
            #     cost_last = 0
            #     for i in range(len(trajectory)):
            #         cost_last += trajectory[i][3]
            #     cost_last += cost
            #     if len(trajectory) >= self.deadline:
            #         reward = 1000 - cost_last
            #     else:
            #         reward = 1000 - cost_last
            #     # reward = 1000 - cost_last
            #     done = 1
            # judge = random.random()
            # if judge<self.P4:
            #     self.state[0] = 7
            # if judge < self.P2:
            #     self.state[0] = 7
            # elif judge > self.P3:
            #     self.state[0] = 9

        # elif self.state[0] == 8 and action == 1:
        #     t_stop = self.M / self.transfer[1] + 1.28
        #     t_2 = self.timeslot - t_stop - self.t_new
        #     self.state[1] -= self.rate[8] * self.t_new
        #     self.state[1] -= self.rate[0] * t_2
        #     #cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
        #     cost = self.uc2 + self.uc1 * self.timeslot
        #     if self.state[1] <= 0:
        #         self.state[1] = 0
        #         cost_last = 0
        #         for i in range(len(trajectory)):
        #             cost_last += trajectory[i][3]
        #         cost_last += cost
        #         if len(trajectory) >= self.deadline:
        #             reward = 1000 - cost_last
            #     else:
            #         reward = 1000 - cost_last
            #     # reward = 1000 - cost_last
            #     done = 1
            # judge = random.random()
            # if judge < self.P1:
            #     self.state[0] = 0
            # else:
            #     self.state[0] = 1
        
        # elif self.state[0] == 9 and action == 0:
        #     cost = self.uc1 * self.timeslot
            # self.state[1] -= self.rate[9] * self.timeslot
            # if self.state[1] <= 0:
            #     self.state[1] = 0
            #     cost_last = 0
            #     for i in range(len(trajectory)):
            #         cost_last += trajectory[i][3]
            #     cost_last += cost
            #     if len(trajectory) >= self.deadline:
            #         reward = 1000 - cost_last
            #     else:
            #         reward = 1000 - cost_last
            #     # reward = 1000 - cost_last
            #     done = 1
            # judge = random.random()
            # if judge<self.P4:
            #     self.state[0] = 8
            # if judge < self.P2:
            #     self.state[0] = 8
            # elif judge > self.P3:
            #     self.state[0] = 10

        # elif self.state[0] == 9 and action == 1:
        #     t_stop = self.M / self.transfer[1] + 1.62
        #     t_2 = self.timeslot - t_stop - self.t_new
        #     self.state[1] -= self.rate[9] * self.t_new
        #     self.state[1] -= self.rate[0] * t_2
        #     #cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
        #     cost = self.uc2 + self.uc1 * self.timeslot
        #     if self.state[1] <= 0:
        #         self.state[1] = 0
        #         cost_last = 0
        #         for i in range(len(trajectory)):
        #             cost_last += trajectory[i][3]
        #         cost_last += cost
            #     if len(trajectory) >= self.deadline:
            #         reward = 1000 - cost_last
            #     else:
            #         reward = 1000 - cost_last
            #     # reward = 1000 - cost_last
            #     done = 1
            # judge = random.random()
            # if judge < self.P1:
            #     self.state[0] = 0
            # else:
            #     self.state[0] = 1
        
        # elif self.state[0] == 10 and action == 0:
        #     cost = self.uc1 * self.timeslot
        #     self.state[1] -= self.rate[10] * self.timeslot
        #     if self.state[1] <= 0:
        #         self.state[1] = 0
        #         cost_last = 0
        #         for i in range(len(trajectory)):
        #             cost_last += trajectory[i][3]
        #         cost_last += cost
            #     if len(trajectory) >= self.deadline:
            #         reward = 1000 - cost_last
            #     else:
            #         reward = 1000 - cost_last
            #     # reward = 1000 - cost_last
            #     done = 1
            # judge = random.random()
            # if judge<self.P4:
            #     self.state[0] = 9
            # if judge < self.P2:
            #     self.state[0] = 9
            # elif judge > self.P3:
            #     self.state[0] = 11

        # elif self.state[0] == 10 and action == 1:
        #     t_stop = self.M / self.transfer[1] + 2
        #     t_2 = self.timeslot - t_stop - self.t_new
        #     self.state[1] -= self.rate[10] * self.t_new
        #     self.state[1] -= self.rate[0] * t_2
        #     #cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
        #     cost = self.uc2 + self.uc1 * self.timeslot
        #     if self.state[1] <= 0:
        #         self.state[1] = 0
        #         cost_last = 0
            #     for i in range(len(trajectory)):
            #         cost_last += trajectory[i][3]
            #     cost_last += cost
            #     if len(trajectory) >= self.deadline:
            #         reward = 1000 - cost_last
            #     else:
            #         reward = 1000 - cost_last
            #     reward = 1000 - cost_last
            #     done = 1
            # judge = random.random()
            # if judge < self.P1:
            #     self.state[0] = 0
            # else:
            #     self.state[0] = 1
        
        # elif self.state[0] == 11 and action == 0:
        #     cost = self.uc1 * self.timeslot
        #     self.state[1] -= self.rate[11] * self.timeslot
        #     if self.state[1] <= 0:
        #         self.state[1] = 0
        #         cost_last = 0
            #     for i in range(len(trajectory)):
            #         cost_last += trajectory[i][3]
            #     cost_last += cost
            #     if len(trajectory) >= self.deadline:
            #         reward = 1000 - cost_last
            #     else:
            #         reward = 1000 - cost_last
            #     # reward = 1000 - cost_last
            #     done = 1
            # judge = random.random()
            # if judge<self.P4:
            #     self.state[0] = 10
            # if judge < self.P2:
            #     self.state[0] = 10
            # elif judge > self.P3:
            #     self.state[0] = 12

        # elif self.state[0] == 11 and action == 1:
        #     t_stop = self.M / self.transfer[1] + 2.42
        #     t_2 = self.timeslot - t_stop - self.t_new
        #     self.state[1] -= self.rate[11] * self.t_new
        #     self.state[1] -= self.rate[0] * t_2
        #     cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
        #     cost = self.uc2 + self.uc1 * self.timeslot
            # if self.state[1] <= 0:
            #     self.state[1] = 0
            #     cost_last = 0
            #     for i in range(len(trajectory)):
            #         cost_last += trajectory[i][3]
            #     cost_last += cost
            #     if len(trajectory) >= self.deadline:
            #         reward = 1000 - cost_last
            #     else:
            #         reward = 1000 - cost_last
                # reward = 1000 - cost_last
            #     done = 1
            # judge = random.random()
            # if judge < self.P1:
            #     self.state[0] = 0
            # else:
            #     self.state[0] = 1
        
        # elif self.state[0] == 12 and action == 0:
        #     cost = self.uc1 * self.timeslot
        #     self.state[1] -= self.rate[12] * self.timeslot
        #     if self.state[1] <= 0:
        #         self.state[1] = 0
        #         cost_last = 0
        #         for i in range(len(trajectory)):
        #             cost_last += trajectory[i][3]
        #         cost_last += cost
        #         if len(trajectory) >= self.deadline:
        #             reward = 1000 - cost_last
        #         else:
        #             reward = 1000 - cost_last
        #         # reward = 1000 - cost_last
        #         done = 1
        #     judge = random.random()
        #     if judge<self.P4:
        #         self.state[0] = 11
        #     # if judge < self.P2:
        #     #     self.state[0] = 7
        #     # elif judge > self.P3:
        #     #     self.state[0] = 9

        # elif self.state[0] == 12 and action == 1:
        #     t_stop = self.M / self.transfer[1] + 2.88
        #     t_2 = self.timeslot - t_stop - self.t_new
        #     self.state[1] -= self.rate[12] * self.t_new
        #     self.state[1] -= self.rate[0] * t_2
        #     #cost = self.uc2 * t_stop + self.uc1 * (self.timeslot - t_stop)
        #     cost = self.uc2 + self.uc1 * self.timeslot
        #     if self.state[1] <= 0:
        #         self.state[1] = 0
        #         cost_last = 0
        #         for i in range(len(trajectory)):
        #             cost_last += trajectory[i][3]
        #         cost_last += cost
        #         if len(trajectory) >= self.deadline:
        #             reward = 1000 - cost_last
        #         else:
        #             reward = 1000 - cost_last
        #         # reward = 1000 - cost_last
        #         done = 1
        #     judge = random.random()
        #     if judge < self.P1:
        #         self.state[0] = 0
        #     else:
        #         self.state[0] = 1

        return ((self.state[0], self.state[1],self.state[2]), reward, done, action, cost)
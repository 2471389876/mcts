#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import random

class Agent:
    def __init__(self):
        # 状态集合包括 ： 距离，数据量，当前时隙
        self.State = self.reset()

    def reset(self):
        return np.array([1, 2000, 0])
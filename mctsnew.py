#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import random
import numpy as np
import Env_8 as Env
env = Env.Environ()
env_train = Env.Environ()
# AVAILABLE_CHOICES = [1, -1, 2, -2]
# AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
# MAX_ROUND_NUMBER = 10

AVAILABLE_CHOICES = [0, 1] #动作集
AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES) #动作长度
trajectory1 = [] #存放模拟扩展的序列

class Node(object):
  """
  蒙特卡罗树搜索的树
  结构的Node，包含了父节点和直接点等信息，还有用于计算UCB的遍历次数和quality值，还有游戏选择这个Node的State。
  """

  def __init__(self):
    self.parent = None
    self.children = []

    self.visit_times = 0
    self.quality_value = 0.0

    self.state = None
    self.action = 2
    self.cost = 0.0

  def set_state(self, state):
    self.state = state

  def get_state(self):
    return self.state

  def set_action(self, action):
    self.action = action

  def get_action(self):
    return self.action

  def set_cost(self, cost):
    self.cost = cost

  def get_cost(self):
    return self.cost

  def get_parent(self):
    return self.parent

  def set_parent(self, parent):
    self.parent = parent

  def get_children(self):
    return self.children

  def get_visit_times(self):
    return self.visit_times

  def set_visit_times(self, times):
    self.visit_times = times

  def visit_times_add_one(self):
    self.visit_times += 1

  def get_quality_value(self):
    return self.quality_value

  def set_quality_value(self, value):
    self.quality_value = value

  def quality_value_add_n(self, n):
    self.quality_value += n

  def is_all_expand(self):
    if(self.state[0]==0):
      return len(self.children) == 2
    elif(self.state[0]==env.maxdistance):
      return len(self.children) == 4
    else:
      return len(self.children) == 3

  def add_child(self, sub_node):
    sub_node.set_parent(self)
    self.children.append(sub_node)

  def __repr__(self):
    return "Node: {}, Q/N: {}/{}, state: {}".format(
        hash(self), self.quality_value, self.visit_times, self.state)


def tree_policy(node):
  """
  蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），根据exploration/exploitation算法返回最好的需要expend的节点，注意如果节点是叶子结点直接返回。

  基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过exploration/exploitation的UCB值最大的，如果UCB值相等则随机选。
  """
  # Check if the current node is the leaf node
  while node.get_state()[1] > 0:
    if node.is_all_expand():
      node = best_child(node, True)
    else:
      # Return the new sub node
      sub_node = expand(node)
      return sub_node

  # Return the leaf node
  return node


def default_policy(node):
  """
  蒙特卡罗树搜索的Simulation阶段，输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward。注意输入的节点应该不是子节点，而且是有未执行的Action可以expend的。

  基本策略是随机选择Action。
  """
  # Get the state of the expand_node
  current_state = node.get_state()
  accumulative_cost = 0
  # Run until the over
  while current_state[1] > 0:
      # Pick one random action to play and get next state
    random_action = random.choice([choice for choice in AVAILABLE_CHOICES]) 
    next_state, reward, done, action, cost = env.step(random_action, trajectory1)
    accumulative_cost += cost
    trajectory1.append((next_state, action, reward, cost))
    current_state = next_state

  # if current_state[2] < env.deadline:
  #   print('success')
  #   final_state_reward = 100
  # else:
  #   final_state_reward = 0
  return accumulative_cost

def expand(node):
  """
  输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
  """
  tried_sub_node_states = [
      sub_node.get_state() for sub_node in node.get_children()
  ]
  random_action = random.choice([choice for choice in AVAILABLE_CHOICES]) 
  env.state[0] = node.get_state()[0]
  env.state[1] = node.get_state()[1]
  env.state[2] = node.get_state()[2]
  next_state, reward, done, action, cost = env.step(random_action, trajectory1)
  new_state = next_state

  
  # Check until get the new state which has the different action from others
  while new_state in tried_sub_node_states:
    random_action = random.choice([choice for choice in AVAILABLE_CHOICES]) 
    env.state[0] = node.get_state()[0]
    env.state[1] = node.get_state()[1]
    env.state[2] = node.get_state()[2]
    next_state, reward, done, action, cost = env.step(random_action, trajectory1)
    new_state = next_state


  trajectory1.append((next_state, action, reward, cost))
  sub_node = Node()
  sub_node.set_state(new_state)
  node.add_child(sub_node)
  sub_node.set_action(action)
  sub_node.set_cost(cost)

  return sub_node


def best_child(node, is_exploration):
  """
  使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是利用阶段直接选择当前Q值得分最高的。
  """

  # TODO: Use the min float value
  best_score = sys.maxsize
  best_sub_node = None

  # Travel all sub nodes to find the best one
  for sub_node in node.get_children():

    # Ignore exploration for inference
    #c是一个常量参数，可以控制探索和利用的权重
    if is_exploration:
      C = 1 / math.sqrt(2.0)
    else:
      C = 0.0
    
    # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
    left = sub_node.get_quality_value() / sub_node.get_visit_times()
    right = 2.0 * math.log(node.get_visit_times()) / sub_node.get_visit_times()
    score = left + C * math.sqrt(right)

    if score < best_score:
      best_sub_node = sub_node
      best_score = score
 
  return best_sub_node


def backup(node, accumulative_cost):
  """
  蒙特卡洛树搜索的Backpropagation阶段，输入前面获取需要expend的节点和新执行Action的reward，反馈给expend节点和上游所有节点并更新对应数据。
  """
  # print('accumulative_cost',accumulative_cost)
  node.quality_value_add_n(accumulative_cost)
  # Update util the root node
  while node != None:
    # Update the visit times
    node.visit_times_add_one()
    # print('visti_time',node.get_visit_times())
    # print('qv',node.get_quality_value())
    # Update the quality value
    if(node.parent):
      value = node.get_quality_value()
      cost = node.get_cost()

    # print('从扩展节点往后的累计',node.get_quality_value())
    # Change the node to the parent node
    node = node.parent
    if(node):
      node.quality_value_add_n(value+cost)
    
 


def monte_carlo_tree_search(node):
  """
  实现蒙特卡洛树搜索算法，传入一个根节点，在有限的时间内根据之前已经探索过的树结构expand新节点和更新数据，然后返回只要exploitation最高的子节点。

  蒙特卡洛树搜索包含四个步骤，Selection、Expansion、Simulation、Backpropagation。
  前两步使用tree policy找到值得探索的节点。
  第三步使用default policy也就是在选中的节点上随机算法选一个子节点并计算reward。
  最后一步使用backup也就是把reward更新到所有经过的选中节点的节点上。

  进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点。
  """

  computation_budget = 500 #限制树搜索次数，实际情况此时还没有找到最好的动作往下执行
  # Run as much as possible under the computation budget
  for i in range(computation_budget):
    # print('从当前节点第',i+1,'次开始扩展')
    # 1. Find the best node to expand 选择加扩展
    expand_node = tree_policy(node)
    # print('父节点为:',expand_node.parent)
    # print('扩展的节点为:',expand_node)
    # print('扩展的节点动作为:',expand_node.get_action())
    
    # 2. Random run to add node and get reward 模拟
    accumulative_cost = default_policy(expand_node)

    # 3. Update all passing nodes with reward 回溯
    backup(expand_node, accumulative_cost)
    # print(trajectory1)
    
  # N. Get the best next node
  best_next_node = best_child(node, False)
  # print('best node',best_next_node)
  best_next_node_parent = best_next_node.parent
 
  env.state[0] = best_next_node_parent.get_state()[0]
  env.state[1] = best_next_node_parent.get_state()[1]
  env.state[2] = best_next_node_parent.get_state()[2]
  nxt_state, reward, done, action, cost = env.step(best_next_node.get_action(),trajectory1)
  
  flag = False
  nxt_node = Node()
  nxt_node.set_state(nxt_state)
  nxt_node.set_cost(cost)

  # 判断新生的子节点是否在父节点里，如果在，则直接赋值，如果不在，则添加到父节点上
  for s_node in best_next_node_parent.get_children():
    if(nxt_node.get_state() == s_node.get_state()):
      flag = True
      nxt_node = s_node
  if(flag == False):
    best_next_node_parent.add_child(nxt_node)

  # print('nxt_node',nxt_node)
  # print('训练之后选择的动作是：',best_next_node.get_action())
  return nxt_node

def cycle():
  # Create the initialized state and initialized node
  init_state = env.reset()
  init_node = Node()
  init_node.set_state(init_state)
  current_node = init_node
  print('init node',current_node)
  cost = 0
  ans = 0
#  从初始态开始执行蒙特卡洛树搜索，循环多少次就选择了多少个节点往下走，也就是多少层
  for i in range(30):
    if(current_node.get_state()[1]>0):
      current_node = monte_carlo_tree_search(current_node)
      # print('根据选择的动作到达的下一个节点',current_node)
      # print('训练之后选择的最好节点',current_node)
      cost += current_node.get_cost()

  if(current_node.get_state()[2]<=env.deadline):
    ans = 1
  print('总cost为：',cost)
  return (cost,ans)


def main():
  C = 0
  Ans = 0
  for i in range(100):
    print('第',i+1,'次遍历')
    all_cost, all_ans = cycle()
    C += all_cost
    Ans += all_ans
  print('average cost:',C/100)
  print('service success ratio',Ans/100)
      
if __name__ == "__main__":
  main()
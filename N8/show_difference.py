import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def smooth(arr):
    smoothing_rate = 0.95
    temp_arr = np.zeros(len(arr))
    temp_arr[0] = arr[0]
    for i in range(1, len(arr)):
        temp_arr[i] = smoothing_rate * temp_arr[i-1] + (1-smoothing_rate) * arr[i]
    return temp_arr

def cal_avg(a):
    r = 0
    for i in range(len(a)):
        r += a[i]
    r = r / len(a)
    return r

if __name__=="__main__":
    x1 = np.loadtxt("data/MCMC_N_8.txt")[0:200]
    x2 = np.loadtxt("data/MFEC_max_N_8.txt")[0:200]
    x3 = np.loadtxt("data/MFEC_avg_N_8.txt")[0:200]
    x4 = np.loadtxt("data/Q_learning_N_8.txt")[0:200]

    s = 180
    e = 200
    print('MCMC,MAX,AVG,QL')
    print(cal_avg(x1[s:e]),cal_avg(x2[s:e]),cal_avg(x3[s:e]),cal_avg(x4[s:e]))

    x1 = smooth(x1)
    x2 = smooth(x2)
    x3 = smooth(x3)
    x4 = smooth(x4)
    iter = range(200)
    sns.set(style="whitegrid", font_scale=1.1)
    sns.tsplot(time=iter, data=x1, color="r",linestyle='-',  condition="MCMC")
    sns.tsplot(time=iter, data=x2, color="b", linestyle='--',condition="MFEC_max")
    sns.tsplot(time=iter, data=x3, color="g",linestyle='-.', condition="MFEC_avg")
    sns.tsplot(time=iter, data=x4, color="gray", linestyle=':', condition="Q_learning")

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 4))
    plt.ylabel("Reward", fontsize=15)
    plt.xlabel("Iteration Number", fontsize=15)
    plt.savefig("data/different_algorithms_deadline.pdf",format='pdf')
    plt.show()






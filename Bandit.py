import pandas as pd
import sys
import random as rd

args = sys.argv[1:]
alg = args[0]
exp = float(args[1])
dist = float(args[2])
decay = float(args[3])
rwt = float(args[4])
w0 = float(args[5])
alpha = decay
weight=[w0,w0,w0,w0,w0,w0]
prob=[1/6,1/6,1/6,1/6,1/6,1/6]
cumulative_reward = 0
infile = pd.read_csv(".\{0}".format(args[6]))

#Normalise the given weights
def Normalise_weight(s):
    min_w = min(s)
    max_w = max(s)
    for i in range(0, len(s)):
        s[i] = (s[i] - min_w) / (max_w - min_w)
    return s

#Normalise the Probabilities
def Normalise_prob(s):
 total  = sum(s)
 for i in range(0,len(s)):
     s[i] = s[i]/total
 return s

#pull variable points to the index of the arm pulled
#reward variable is for the current reward earned
#cumulative_reward is for the total reward earned till that step

#STAT algorithm
if alg == "STAT":
    for i in range(0,len(infile)):
      prob = Normalise_prob(prob)
      j = rd.choices(prob)
      pull = prob.index(j[0])
      print("\nStep {0}".format(i+1))
      print("Bandit {0} is pulled".format(pull+1))
      reward = infile.iloc[i,pull]
      print("Reward obtained: {0}".format(reward))
      cumulative_reward = cumulative_reward + reward
      print("Cumulative Reward: {0}".format(cumulative_reward))
      weight[pull] = decay*weight[pull] + rwt*reward
      norm_weight = Normalise_weight(weight.copy())
      prob[pull] = (norm_weight[pull]*(1-exp)) + (exp*dist)


#times[] list is used to store the number of times an arm is pulled
#ROLL algorithm
if alg == "ROLL":
    times=[0,0,0,0,0,0]
    for i in range(0,len(infile)):
        prob = Normalise_prob(prob)
        j = rd.choices(prob)
        pull = prob.index(j[0])
        print("\nStep {0}".format(i + 1))
        times[pull]=times[pull]+1
        print("Bandit {0} is pulled".format(pull+1))
        reward = infile.iloc[i, pull]
        cumulative_reward = cumulative_reward + reward
        print("Cumulative Reward: {0}".format(cumulative_reward))
        print("Reward: {0}".format(reward))
        weight[pull] = weight[pull] + (reward-weight[pull])/(times[pull])
        norm_weight = Normalise_weight(weight.copy())
        prob[pull] = (norm_weight[pull]*(1-exp)) +(exp*dist)

#Sigma part of equation in REC algorithm

def summation(length,rlist):
 sum = 0
 for i in range(0,length):
  sum=sum+(alpha*(pow((1-alpha),length-i)*rlist[i]))
 return sum

#REC algorithm
#Weight_list[] stores the list of weights for each arm
#reward_list[] stores the list of rewards for each arm

if alg == "REC":
    weight_list = [[],[],[],[],[],[]]
    reward_list = [[],[],[],[],[],[]]
    for i in range(0,len(infile)):
        prob = Normalise_prob(prob)
        j = rd.choices(prob)
        pull = prob.index(j[0])
        print("\nStep {0}:".format(i+1))
        print("Bandit {0} is pulled".format(pull+1))
        reward = infile.iloc[i,pull]
        cumulative_reward = cumulative_reward + reward
        print("Reward obtained: {0}".format(reward))
        print("Cumulative Reward: {0}".format(cumulative_reward))
        length = len(weight_list[pull])
        if length >= 1:
         weight_list[pull].append(pow((1-alpha),length)*weight_list[pull][0]+summation(length,reward_list[pull].copy()))
         norm_weight = Normalise_weight(weight_list[pull].copy())
         prob[pull] = (norm_weight[length] * (1 - exp)) + (exp * dist)
        else:
         weight_list[pull].append(w0)
         prob[pull] = (w0*(1-exp)) + (exp*dist)
        reward_list[pull].append(reward)
        if length == 10:
         weight_list[pull].pop(0)











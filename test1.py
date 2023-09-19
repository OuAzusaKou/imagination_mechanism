import random
import numpy as np
def cal_pi(sim_num):
    inside_num=0
    for i in range(sim_num):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        dis = x**2+y**2

        if dis <= 1:
            inside_num+=1
            
    value = 4 * float(inside_num/sim_num)
    return value

print(cal_pi(100000))
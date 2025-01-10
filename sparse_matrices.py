# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:51:52 2025

@author: barba
"""
import time
import itertools
# import cProfile
# import pstats
import numpy as np
# import matplotlib.pyplot as plt
from scipy.sparse import coo_array, eye, linalg
#%%
def transition(state, transition, params, servers_not_full):
    min_queue = -1
    new_state = state
    mus = params[1:]
    div = state[:-1] / mus
    if transition == 0:  # arrival
        if servers_not_full[0]==True and servers_not_full[1]==True:
            min_queue = 0 if div[0] <= div[1] else 1  # wJSQ
        elif servers_not_full[0]==False:
            min_queue = 1
        elif servers_not_full[1] == False:
            min_queue = 0
        new_state[min_queue] += 1
        if min_queue == 0:
            new_state[-1] += 1
            
    elif transition == 1:  # finished service at S1
        if servers_not_full[2]==True and servers_not_full[3]==True:
            min_queue = 2 if div[2] <= div[3] else 3
        elif servers_not_full[2]==False:
            min_queue = 3
        elif servers_not_full[3] == False:
            min_queue = 2
        new_state[0] -= 1
        new_state[min_queue] += 1
       
    
        # new_state[0] -= 1
        # min_queue = 2 if div[2] <= div[3] else 3
        # new_state[min_queue] += 1
        # #new_state[-1] = -2 if min_queue ==2 else -3
    elif transition == 2:  # finished service at S2
        if servers_not_full[3]==True and servers_not_full[4]==True:
            min_queue = 3 if div[3] <= div[4] else 4
        elif servers_not_full[3] == False:
            min_queue = 4
        elif servers_not_full[4] == False:
            min_queue = 3
    
        new_state[1] -= 1
        new_state[min_queue] += 1

        # new_state[1] -= 1
        # min_queue = 3 if div[3] <= div[4] else 4
        # new_state[min_queue] += 1
        # #new_state[-1] = -3 if min_queue ==3 else -4
    else:  # other transitions
        new_state[transition - 1] -= 1
    return new_state, min_queue
#%%

start = time.time()
params = np.array([1,1,1,2,1,1])
N= 3
states = list(itertools.product(range(0,N+1), range(N), range(2*N), range(2*N), range(2*N), range(N)))
#states =  [s for s in states if s[0]-s[-1]>0]
nr_states =  len(states)
it = 0
row = []
col = []
data = []
it = 0
b = []
print(f"need to do {len(states)} iterations")
#%%
for state in states:
    if state[0] - state[-1]<0:
        b.append(0)
    elif state[0] - state[-1]==0:
        b.append(np.min([state[2]/params[3], state[3]/params[4]]))
    else:

        
        s1_not_full = state[0] <= min(5, N-1)
        s2_not_full = state[0] <= min(5, N-2)
        s3_not_full = state[2] <= 2*N-2
        s4_not_full = state[3] <= 2*N-2
        s5_not_full = state[4] <= 2*N-2
        
        servers_not_full = [s1_not_full, s2_not_full, s3_not_full, s4_not_full, s5_not_full]
        
        active_transitions = [
            s1_not_full or s2_not_full,  # Arrival
            state[0] > 0 and (s3_not_full or s4_not_full),  # service at Q1
            state[1] > 0 and (s4_not_full or s5_not_full),  # service at Q2
            state[2] > 0,  # service at Q3
            state[3] > 0,  # service at Q4
            state[4] > 0,  # service at Q5
            ]
        asum = np.dot(active_transitions , params)
        probs = active_transitions * params / asum
        C = 1/asum
        b.append(C)
        for i in range(6):
            if (not active_transitions[i]):
                continue
            
            new_state, min_queue = transition(list(state), i, params, servers_not_full)
            
            if new_state[0] == new_state[-1]: #tagged customer finished service at S1
                if min_queue == 2:
                    b[it] += probs[i]*state[2]/params[3]
                elif min_queue == 3:
                    b[it] += probs[i] *state[3]/params[4]
            else:
                idx = states.index(tuple(new_state))
                row.append(it)
                col.append(idx)
                data.append(probs[i])
            if np.any(np.array(new_state)==-1): 
                print(state, active_transitions,i, new_state)
        active_transitions = np.array(active_transitions)

    it += 1
        
    if it%(nr_states//20) ==0:
        print(it)

    #     idx = states.index(new_state)
    #     new_states_indices.append(idx)
        
    # it += 1
    # if it%1000==0:
    #     print(it)

# for s in [i for i in enum_states if i[0]<20]:
#     print(s[0], s[1])
end=time.time()

print(f"making the matrix took {end-start:.2f} seconds")

start = time.time()
rows = np.array(row); cols = np.array(col)
datas = np.array(data)

A=coo_array((datas, (rows, cols)), shape=(nr_states,  nr_states)).tocsr()
v = np.transpose(np.array(b));
I = eye(m = nr_states, n = nr_states)
M = I-A
wait_times = linalg.spsolve(M,v)
end = time.time()
print(f"solving the system took {end-start:.2f} seconds")


print(states.index((1,0,1,3,0,0)))
print(wait_times[states.index((1,0,1,3,0,0))])
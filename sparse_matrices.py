# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:51:52 2025

@author: barba
"""
import time
import itertools
import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_array, eye, linalg, save_npz, load_npz, triu, tril, coo_matrix
import matspy
from numba import njit

import os
os.chdir("C:\\Users\\barba\\Desktop\\thesis\\python")

"""
remove the all zero rows, keep track of these indices when creating a matrix

analytical way to speed it up? split into subproblems tha thave easy solutions, such as diagonal
metrices which can be easily inverted?

remove the states where you know the solution, such as 1x0xx,0, 2x0xx0?
"""



#%%

def transition(state, transition, params, servers_not_full):
    
    min_queue = -1
    new_state = state.copy()
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
       
    elif transition == 2:  # finished service at S2
        if servers_not_full[3]==True and servers_not_full[4]==True:
            min_queue = 3 if div[3] <= div[4] else 4
        elif servers_not_full[3] == False:
            min_queue = 4
        elif servers_not_full[4] == False:
            min_queue = 3
    
        new_state[1] -= 1
        new_state[min_queue] += 1


    else:  # other transitions
        new_state[transition - 1] -= 1
        
    return new_state, min_queue

def ind(state, N):
    [n1,n2,n3,n4,n5,i] = state
    idx = n1*N**2*(N+10)**3 + n2*N*(N+10)**3 +n3*N*(N+10)**2 + n4*N*(N+10) +n5*N +i
    return idx


def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m.col, m.row, 's', color='black', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')

    ax.invert_yaxis()
    ax.set_aspect('equal')

    return ax


def main(N, params):
    start = time.time()
    states = list(itertools.product(np.arange(N+1), np.arange(N), np.arange(N+10), np.arange(N+10), np.arange(N+10), np.arange(N)))
    nr_states =  len(states)
    it = 0
    row = []
    col = []
    data = []
    it = 0
    b = []
    print(f"need to do {len(states)} iterations")
    valid_indices = []
    progress = 0
    
    for state in states:
        
        if state[0] - state[-1]<0:
            b.append(0)
        elif state[0] - state[-1]==0:
            b.append(np.min([state[2]/params[3], state[3]/params[4]]))
        else:
    
            valid_indices.append(it)
            s1_not_full = state[0] <= 5
            s2_not_full = state[0] <= 5
            s3_not_full = state[2] <= N+8
            s4_not_full = state[3] <= N+8
            s5_not_full = state[4] <= N+8
            
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
            for j in range(6):
                if (not active_transitions[j]):
                    continue
              
                new_state, min_queue = transition(list(state), j, params, servers_not_full, s)
                    
                if new_state[0] == new_state[-1]: #tagged customer finished service at S1
                    if min_queue == 2:
                        b[it] += probs[j]*state[2]/params[3]
                    elif min_queue == 3:
                        b[it] += probs[j] *state[3]/params[4]
                else:
                    row.append(it)
                    col.append(ind(new_state, N))
                    data.append(probs[j])
                if np.any(np.array(new_state)==-1): 
                    print(state, active_transitions,j, new_state)
            active_transitions = np.array(active_transitions)
            
                
        it += 1
            
        if it%(nr_states//10) ==0:
            progress += 10
            print(f"progress: {progress} %, time: {time.time()-start:.2f} seconds")
    
    
    rows = np.array(row); cols = np.array(col)
    datas = np.array(data)

    A=coo_array((datas, (rows, cols)), shape=(nr_states,  nr_states)).tocsr()

    v = np.transpose(np.array(b));
    I = eye(m = nr_states, n = nr_states)
    M = I-A


    M = M[valid_indices, :]
    M = M[:, valid_indices]
    v = v[valid_indices]
    
    end=time.time()
    print(f"for threshold N={N}, making the matrix took {end-start:.2f} seconds")
    return M,v

# wrapper function for convenience
def interpol(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)

#%%

N = 10
params = np.array([1,1,1,1,1,1])

M,v = main(N,params)

l=0
u=0

# for d in range(np.shape(M)[0]-1):
#     if np.any(M.diagonal(d)!=0):
#         u += 1
#     if np.any(M.diagonal(-d)!=0):
#         l +=1        
#%%
start = time.time()
wait_times1 = linalg.lsqr(M,v, show=True, atol = 1e-10, btol=1e-10)
end = time.time()
print(f"solving the system for N={N} took {end-start:.2f} seconds\n")
#%%

name = str(params) + ".N+10"
np.save('obs_wt/'+name, wait_times[0])




#%%
# real_states = [s for s in states if s[0]-s[-1]>=1]
# state = (1,0,1,6,0,0); state_ind = real_states.index(state)
# print(wait_times[state_ind])
# print(M[[state_ind] , :].nonzero())
# #%%
# M=M.tocoo()
# plot_coo_matrix(M)

# #%%

# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()
#     wt = main( [2, 2, 1, 0, 1, 0],np.array([0.1, 1., 1., 2., 2., 2.]))    

#     profiler.disable()

#     # display the profiling results
#     stats = pstats.Stats(profiler)
#     stats.sort_stats('time').print_stats(10)  # Top 10 results sorted by time
 
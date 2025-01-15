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
import pickle
import os
import matplotlib
os.chdir("C:\\Users\\barba\\Desktop\\thesis\\python")

"""
remove the states where you know the solution, such as 1x0xx,0, 2x0xx0?
"""



#%%

def transition1(state, transition, params, servers_not_full,policy):
    min_queue = -1
    new_state = state.copy()
    mus = params[1:]
    div = state[:-1] / mus
    if transition == 0:  # arrival
        if servers_not_full[0]==True and servers_not_full[1]==True:
            min_queue = policy[tuple(state[:-1])]
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


def transition2(state, transition, params, servers_not_full, policy):
    min_queue = -1
    new_state = state.copy()
    mus = params[1:]
    div = state[:-1] / mus
    if transition == 0:  # arrival
        if servers_not_full[0]==True and servers_not_full[1]==True:
            min_queue = policy[tuple(state[:-1])]
        elif servers_not_full[0]==False:
            min_queue = 1
        elif servers_not_full[1] == False:
            min_queue = 0
        new_state[min_queue] += 1
        if min_queue == 1:
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




def ind1(state, Ns):
    [N1,N2,N3,N4,N5] = Ns
    [n1,n2,n3,n4,n5,i] = state
    idx = n1*N1*N5*N4*N3*N2 + n2*N1*N5*N4*N3 +n3*N1*N5*N4 + n4*N1*N5 +n5*N1 +i
    return idx

def ind2(state, Ns):
    [N1,N2,N3,N4,N5] = Ns
    [n1,n2,n3,n4,n5,i] = state
    idx = n1*N2*N5*N4*N3*(N2+1) + n2*N2*N5*N4*N3 +n3*N2*N5*N4 + n4*N2*(N5) +n5*N2 +i
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


def solve_system2(Ns, params, policy):
    N1,N2,N3,N4,N5 = Ns
    states2 = list(itertools.product(np.arange(N1), np.arange(N2+1), np.arange(N3), np.arange(N4), np.arange(N5), np.arange(N2)))
    nr_states2 =  len(states2)
    it = 0
    row = []
    col = []
    data = []
    b = []

    valid_indices = []
    progress = 0
    
    for state in states2:
        if state[1] - state[-1]<=0:
            b.append(0)
        else:
            valid_indices.append(it)
            s1_not_full = state[0] <= N1-2
            s2_not_full = state[1] <= N2-1
            s3_not_full = state[2] <= N3-2
            s4_not_full = state[3] <= N4-2
            s5_not_full = state[4] <= N5-2
            
            servers_not_full = [s1_not_full, s2_not_full, s3_not_full, s4_not_full, s5_not_full]
            
            active_transitions = [
                s1_not_full or s2_not_full, #and state[-1]<= 5,  # Arrival
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
                
                new_state, min_queue = transition2(list(state), j, params, servers_not_full, policy)
                    
                if new_state[1] == new_state[-1]: #tagged customer finished service at S1
                    if min_queue == 3:
                        b[it] += probs[j]*(new_state[3]-1)/params[4]
                    elif min_queue == 4:
                        b[it] += probs[j] *(new_state[4]-1)/params[5]
                else:
                    row.append(it)
                    col.append(ind2(new_state, Ns))
                    data.append(probs[j])
            active_transitions = np.array(active_transitions)
            
                
        it += 1
            
        if it%(nr_states2//10) ==0:
            progress += 10
            print(f"progress: {progress} %")
    

    
    rows = np.array(row); cols = np.array(col)
    datas = np.array(data)

    A=coo_array((datas, (rows, cols)), shape=(nr_states2,  nr_states2)).tocsr()

    v = np.transpose(np.array(b));
    I = eye(m = nr_states2, n = nr_states2)
    M = I-A


    M = M[valid_indices, :]
    M = M[:, valid_indices]
    v = v[valid_indices]
    wait_times2 = linalg.lsqr(M,v, atol = 1e-10, btol=1e-10)[0]
    
    if np.any(wait_times2<-0.1):
        print("negative solutions")
    
    #wait_times2[wait_times2<0] = 0
    states2 = [s for s in states2 if s[1]-s[-1]>0]
    wait_times_dict = dict(zip(states2,wait_times2))
    wait_times_dict = {key: value for key, value in wait_times_dict.items() if key[-1] == 0}

    return wait_times_dict, M

def solve_system1(Ns, params, policy):
    [N1,N2,N3,N4,N5] = Ns

    states1 = list(itertools.product(np.arange(N1+1), np.arange(N2), np.arange(N3), np.arange(N4), np.arange(N5), np.arange(N1)))
    nr_states1 =  len(states1)
    it = 0
    row = []
    col = []
    data = []
    b = []
    valid_indices = []
    progress = 0
    
    for state in states1:
        if state[0] - state[-1]<=0:
            b.append(0)
        else:
            
            valid_indices.append(it)
            s1_not_full = state[0] <= N1-1 
            s2_not_full = state[1] <= N2-2 
            s3_not_full = state[2] <= N3-2
            s4_not_full = state[3] <= N4-2
            s5_not_full = state[4] <= N5-2
            
            servers_not_full = [s1_not_full, s2_not_full, s3_not_full, s4_not_full, s5_not_full]
            
            
            
            active_transitions = [
                s1_not_full or s2_not_full, #and state[-1]<= 5,  # Arrival
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
                
                new_state, min_queue = transition1(list(state), j, params, servers_not_full, policy)
                if new_state[0] == new_state[-1]: #tagged customer finished service at S1
                    if min_queue == 2:
                        b[it] += probs[j]*(new_state[2]-1)/params[3]
                    elif min_queue == 3:
                        b[it] += probs[j] *(new_state[3]-1)/params[4]
                else:
                    row.append(it)
                    col.append(ind1(new_state, Ns))
                    data.append(probs[j])
            active_transitions = np.array(active_transitions)
            
                
        it += 1
            
        if it%(nr_states1//10) ==0:
            progress += 10
            print(f"progress: {progress} %")
    
    
    rows = np.array(row); cols = np.array(col)
    datas = np.array(data)

    A=coo_array((datas, (rows, cols)), shape=(nr_states1,  nr_states1)).tocsr()

    v = np.transpose(np.array(b));
    I = eye(m = nr_states1, n = nr_states1)
    M = I-A


    M = M[valid_indices, :]
    M = M[:, valid_indices]
    v = v[valid_indices]
    wait_times1 = linalg.lsqr(M,v, atol = 1e-10, btol=1e-10)[0]
    

    if np.any(wait_times1<-0.1):
        print("negative solutions")

    #wait_times1[wait_times1<0] = 0

    

    states1 = [s for s in states1 if s[0]-s[-1]>0]
    wait_times_dict = dict(zip(states1,wait_times1))
    wait_times_dict = {key: value for key, value in wait_times_dict.items() if key[-1] == 0}

    return wait_times_dict, M



#%%

start = time.time()
N=7
Ns = [N,N,N,N,N]
params = np.array([1., 3., 3., 1., 1., 1.])

policy= {}

[N1,N2,N3,N4,N5] = Ns
states = list(itertools.product(np.arange(N1), np.arange(N2), np.arange(N3), np.arange(N4), np.arange(N5)))

for s in states:
    s1 = list(s).copy(); s1[0]+=1
    s2 = list(s).copy(); s2[1]+=1
    policy[s] = 0 if s1[0]/params[1] <= s2[1]/params[2] else 1

for i in range(10):
    print(f"i = {i}")
    time1 = time.time()
    wait_times1, M1 = solve_system1(Ns,params,policy)
    time2 = time.time()
    print(f"time = {time2-time1:.2f} seconds")
    wait_times2, M2 = solve_system2(Ns,params,policy)
    time3 = time.time()
    print(f"time = {time3-time2:.2f} seconds")

    wait_times1dict = wait_times1
    wait_times2dict = wait_times2
    
    new_policy= {}
    
    for s in states:
        s1 = list(s).copy(); s1[0]+=1; s1.append(0)
        s2 = list(s).copy(); s2[1]+=1; s2.append(0)
        if wait_times1[tuple(s1)] <= wait_times2[tuple(s2)]:
            new_policy[s] = 0
        else:
            new_policy[s] = 1
    dis = [s for s in states if new_policy[s]!=policy[s]]
    
    print(len(dis))
    
    if len(dis) == 0:
        break
    
    policy = new_policy
    
end = time.time()

print(f"in total this took {(end-start)/60:.2f} minutes")


#%%
# real_states = [s for s in states if s[0]-s[-1]>=1]
# state = (1,0,1,6,0,0); state_ind = real_states.index(state)
# print(wait_times[state_ind])
# print(M[[state_ind] , :].nonzero())
# #%%

# #%%

# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()
#     wt = main( [2, 2, 1, 0, 1, 0],np.array([0.1, 1., 1., 2., 2., 2.]))    

#     profiler.disable()

#     # display the profiling results
#     stats = pstats.Stats(profiler)
#     stats.sort_stats('time').print_stats(10)  # Top 10 results sorted by time
 #%%
N=10
n1r = np.array(range(N))
n2r = np.array(range(N))
Y,X = np.meshgrid(n1r, n2r)

# plt.scatter(X[best_resp==2], Y[best_resp==2], color='red', label='Route 2-4', marker=matplotlib.markers.MarkerStyle('o', fillstyle = 'top'))
# plt.scatter(X[best_resp==1], Y[best_resp==1], color='blue', label='Route 1-3', marker=matplotlib.markers.MarkerStyle('o', fillstyle = 'top'))
# plt.scatter(X[best_resp==1.5], Y[best_resp==1.5], color='yellow', label='Both routes optimal',marker=matplotlib.markers.MarkerStyle('o', fillstyle = 'top'))
#%%

N=10
n1r = np.array(range(N))
n2r = np.array(range(N))

pol = [[k[:2],v] for k,v in policy.items() if k[2:]==(0,0,0)]

x_coords = [item[0][0] for item in pol]
y_coords = [item[0][1] for item in pol]
colors = ['red' if item[1] == 1 else 'blue' if item[1]==0 else "green" for item in pol]

plt.figure(figsize=(6, 6))
plt.scatter(x_coords, y_coords, c=colors, s=100)
plt.title("Nash strategy $n_3=1,n_4=n_5=5$, all rates equal to 1")
plt.xlabel("$n_1$")
plt.ylabel("$n_2$")

plt.scatter([], [], c='red', label='$S_2$')
plt.scatter([], [], c='blue', label='$S_1$')
plt.legend(title="Value", loc="upper right")

plt.grid(True)
plt.xticks(range(8))
plt.yticks(range(8))
plt.show()

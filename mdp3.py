import heapq
import time
from dataclasses import dataclass
import itertools
# import cProfile
# import pstats
import pickle
# import numpy as np
# import matplotlib.pyplot as plt
#%%
"""

impossible to see N-1 people at second stage queues? why?
check how much we lose by 

"""

@dataclass
class State:
    params: list  # [n1, n2, n3, n4, n5, i] with i = number of customers behind us in Q1
    prob: float  # likelihood of state
    #label: int # label of newly encountered states (l>0) used to track a state through iteration of main, and update
    # its pi3, pi4. i=0 when a state has been seen before

def main(state, rates, wait_times_dictionary, N):
    n1, n2, n3, n4, n5, dist_end = state
    lam, mu1, mu2, mu3, mu4, mu5 = rates
    pi3 = [0 for i in range(N)]
    pi4 = [0 for i in range(N)]
    
    progress_dict = {}
    heap = []
    root = State([n1, n2, n3, n4, n5, dist_end], 1.0    )
    push(heap, root)
    i = 1; pi_sum = 0
    while pi_sum < 0.97:
        if heap:
            pt = time.time()
            _, _, best_node = heapq.heappop(heap)
            expand_heap(heap, best_node, rates, wait_times_dictionary, pi3, pi4, n1,n2,n3,n4,n5, dist_end, N)
            tt =  time.time()
        i += 1
        pi_sum = sum(pi3) + sum(pi4)
        progress_dict[i] = pi_sum
    
    
    
    pi_sum = sum(pi3) + sum(pi4)
    pi3 = [pi3[i] / pi_sum for i in range(len(pi3))]
    pi4 = [pi4[i] / pi_sum for i in range(len(pi4))]
    
    wt = (
        (n1 - dist_end -1)/ mu1 +
        + sum(pi3[i]*i/mu3 for i in range(len(pi3)))
        + sum(pi4[i]*i/mu4 for i in range(len(pi4)))
    )
    return wt, pi3, pi4, i, progress_dict


def expand_heap(heap, elem, rates, wait_times_dictionary, pi3, pi4, n1,n2,n3,n4, n5, dist_end, N):
    serv1busy = elem.params[0] > 0
    serv2busy = elem.params[1] > 0
    serv34capacity = (elem.params[2]<=N-2 or elem.params[3]<=N-2)
    serv45capacity = (elem.params[3]<=N-2 or elem.params[4]<=N-2)
    active_transitions = [
        (elem.params[0] < 5 or elem.params[1]<5),  # Arrival
        serv1busy and serv34capacity,  # service at Q1
        serv2busy and serv45capacity,  # service at Q2
        elem.params[2] > 0,  # service at Q3
        elem.params[3] > 0,  # service at Q4
        elem.params[4] > 0,  # service at Q5
    ]
    asum = sum(rates[i] for i, active in enumerate(active_transitions) if active)
    probs = [rates[i] / asum if active else 0 for i, active in enumerate(active_transitions)]
    
    # print("\nelement expanded is", elem.params)
    # print("active transitions are", active_transitions)
    # print("transitions start:")
    for i in range(6):
        if not active_transitions[i]:
            continue
        new_state, min_queue = transition(elem.params, i, rates)
        prob_state = elem.prob * probs[i]
        terminal = new_state[0] == new_state[-1]
        if terminal:
            if min_queue == 2:
                pi3[new_state[2] - 1] += prob_state
            elif min_queue == 3:
                pi4[new_state[3] - 1] += prob_state
            continue

        state_seen_before = tuple(new_state) in wait_times_dictionary.keys()
        
        if state_seen_before:
            pi3_old, pi4_old = wait_times_dictionary[tuple(new_state)][1:]
            if sum(pi3_old)+sum(pi4_old)>=0.99:
                for j in range(N):
                    pi3[j] +=  prob_state * pi3_old[j]
                for j in range(N):
                    pi4[j] += prob_state * pi4_old[j]
            else:    
                push(heap, State(new_state, prob_state))
        else:
            push(heap, State(new_state, prob_state))


def transition(state, transition, rates):
    min_queue = -1
    new_state = state.copy()
    div = [0,0,0,0,0,0]
    for i in range(5):
        div[i] = state[i]/rates[i+1]
    
    if transition == 0:  # arrival
        min_queue = 0 if div[0] <= div[1] else 1  # wJSQ
        new_state[min_queue] += 1
        if min_queue == 0:
            new_state[-1] += 1
    elif transition == 1:  # finished service at S1
         new_state[0] -= 1
         min_queue = 2 if div[2] <= div[3] else 3
         new_state[min_queue] += 1
    elif transition == 2:  # finished service at S2
         new_state[1] -= 1
         min_queue = 3 if div[3] <= div[4] else 4
         new_state[min_queue] += 1      
    else:  # other transitions
        new_state[transition - 1] -= 1
    return new_state, min_queue


def push(heap, nd: State):
    heapq.heappush(heap, (-nd.prob, id(nd), nd))
    

#%%
N= 8
states = list(itertools.product(range(1,N+1), range(N), range(N), range(N), range(N), range(N)))
avail_states =  [s for s in states if s[0]-s[-1]>0 and  s[0]-s[-1]>1]

states8 = [s for s in states if s[0]-s[-1]==1 and s[0]==8]
states7 = [s for s in states if s[0]-s[-1]==1 and s[0]==7]
states6 = [s for s in states if s[0]-s[-1]==1 and s[0]==6]
states5 = [s for s in states if s[0]-s[-1]==1 and s[0]==5]
states4 = [s for s in states if s[0]-s[-1]==1 and s[0]==4]
states3 = [s for s in states if s[0]-s[-1]==1 and s[0]==3]
states2 = [s for s in states if s[0]-s[-1]==1 and s[0]==2]
states1 = [s for s in states if s[0]-s[-1]==1 and s[0]==1]

states = [*states8,*states7,*states6,*states5,*states4, *states3, *states2, *states1, *avail_states]

#%%
start = time.time()
i = 1
wait_times1 = {}
times = {}

total_iterations = len(states)
print(f"total number of iterations needed: {total_iterations}")
parameters= [1,1,1,1,1,1]
#%%
for state in states:
    pt = time.time()
    wt, pi3, pi4, niter,_ = main(state,parameters, wait_times1, N)
    if i%1500==0:
        print(f"i = {i},current state is {state}, cummulative time = {pt-start:2f} seconds, niter={niter}")
    tt = time.time()
    wait_times1[tuple(state)] = [wt,pi3,pi4]
    times[tuple(state)] = tt-pt
    i += 1
end = time.time()
print(f"Total time:{end-start:.2f} seconds")
#%%
# states = list(itertools.product(range(1,N+1), range(N), range(N), range(N), range(N), range(N)))
# avail_states =  [s for s in states if s[0]-s[-1]>0 and  s[0]-s[-1]>1]
# for i in range(N):
#     a = [s for s in states if s[0]-s[-1]==1 and s[0]==i]
#     states = [*a, *states]
#%%
name = "wait_times" + str(parameters)+"mark.pkl"
with open(name, 'wb') as f:
    pickle.dump(wait_times1, f)
name = "times" + str(parameters)+"mark.pkl"
with open(name, 'wb') as f:
    pickle.dump(times, f)
    
name = "wait_times" + str(parameters)+"mark.pkl"
with open(name, 'rb') as f:
    load_wait_times1 = pickle.load(f)

#name = "times" + str(parameters)+"mark.pkl"
# with open(name, 'rb') as f:
#     load_times = pickle.load(f)
    
    
#%%

# _,_,_,_,d = main([2, 0, 7, 0,0,0], parameters, wait_times1, N)
#%%
# i = list(d.keys())
# i = np.array(i)
# sums = np.array(list(d.values()))

# plt.plot(i,sums)

#%%


# wait_times1 = {}
# i = 1
# start = time.time()
# for state in itertools.product(range(1,6), range(5), range(5), range(5), range(5), range(5)):
    
#     if state[0]-state[-1]>0:
#         wt, pi3, pi4 = main(state,[0.1,1,0.1,2,2,2], wait_times1)
#         wait_times1[tuple(state)] = [wt,pi3,pi4]
#         if i%500 == 0:
            
#             print(f"i = {i}, time = {time.time()-start :.2f} seconds")
#             print(f"current state is {state}")
#         i += 1

    
# end = time.time()
# print(f"DONE!Total time:{end-start} seconds\n")


#%%



# states = list(itertools.product(range(1,6), range(5), range(5), range(5), range(5), range(5)))
# states =  [s for s in states if s[0]-s[-1]>0]
# almost_terminal_states = [s for s in states if s[0]-s[-1]<=2]

# states = [s for s in states if not s in almost_terminal_states]

# almost_terminal_states.extend(states)

# wait_times1 = {}
# i = 1

# start = time.time()
# total_iterations = len(almost_terminal_states)
# print(f"total number of iterations needed: {total_iterations}")
# for state in almost_terminal_states:
#     wt, pi3, pi4 = main(state,[0.01,1,0.1,2,2,2], wait_times1)
#     wait_times1[tuple(state)] = [wt,pi3,pi4]
#     if i%500 == 0:
        
#         print(f"i = {i}, time = {time.time()-start :.2f} seconds")
#         print(f"current state is {state}")
#     i += 1

# end = time.time()
# print("Total time:", end - start)

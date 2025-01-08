import heapq
import time
from dataclasses import dataclass
import itertools
import cProfile
import pstats
import array
import pickle
#%%

@dataclass
class State:
    params: list  # [n1, n2, n3, n4, n5, i] with i = number of customers behind us in Q1
    prob: float  # likelihood of state

def main(state, rates, wait_times_dictionary, N):
    n1, n2, n3, n4, n5, dist_end = state
    lam, mu1, mu2, mu3, mu4, mu5 = rates
    pi3 = [0 for i in range(n1 + n3 - dist_end)]
    pi4 = [0 for i in range(n1 + n2 + n4 + 5 - dist_end)]
    
    heap = []
    root = State([n1, n2, n3, n4, n5, dist_end], 1.0)
    push(heap, root)
    i = 1
    while sum(pi3) + sum(pi4) < 0.99:
        if heap:
            _, _, best_node = heapq.heappop(heap)
            expand_heap(heap, best_node, rates, wait_times_dictionary, pi3, pi4, n1,n2,n3,n4,n5, dist_end, N)
        i += 1

    pi_sum = sum(pi3) + sum(pi4)
    pi3 = [pi3[i] / pi_sum for i in range(len(pi3))]
    pi4 = [pi4[i] / pi_sum for i in range(len(pi4))]
    

    wt = (
        (n1 - dist_end -1)/ mu1 +
        + sum(pi3[i]*i/mu3 for i in range(len(pi3)))
        + sum(pi4[i]*i/mu4 for i in range(len(pi4)))
    )
    return wt, pi3, pi4


def expand_heap(heap, elem, rates, wait_times_dictionary, pi3, pi4, n1,n2,n3,n4, n5, dist_end, N):
    active_transitions = [
        (elem.params[0] < 5 or elem.params[1]<5),  # Arrival
        elem.params[0] > 0 and (elem.params[2]<=N-2 or elem.params[3]<=N-2),  # service at Q1
        elem.params[1] > 0 and (elem.params[3]<=N-2 or elem.params[4]<=N-2),  # service at Q2
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
        #print(f"transition is {i}, new_state is {new_state}")
        if terminal:
            if min_queue == 2:
                pi3[new_state[2] - 1] += prob_state
            elif min_queue == 3:
                pi4[new_state[3] - 1] += prob_state
            continue

        state_seen_before = tuple(new_state) in wait_times_dictionary.keys()
        #print(f"state seen before: {state_seen_before}")
        if state_seen_before:
            pi3_old, pi4_old = wait_times_dictionary[tuple(new_state)][1:]

            l1 = len(pi3)-len(pi3_old)
            l2 = len(pi4)-len(pi4_old)
            
            if l1>0:
                pi3_old.extend([0 for i in range(l1)])
            elif l1<0:
                pi3_old = pi3_old[:len(pi3)]
            
            if l2>0:
                pi4_old.extend([0 for i in range(l2)])
            elif l2<0:
                pi4_old = pi4_old[:len(pi4)]

            
            for j in range(len(pi3_old)):
                pi3[j] +=  prob_state * pi3_old[j]
            for j in range(len(pi4_old)):
                pi4[j] += prob_state * pi4_old[j]
            
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
    
    
#   elif transition == 1 or transition ==2:
      # t1 = transition + 1; t2 = transition+2
      # new_state[transition-1] -= 1
      # min_queue = t1 if div[t1]<= div[t2] else t2
      # new_state[min_queue] += 1
#elif transition == 1:  # finished service at S1
#     new_state[0] -= 1
#     min_queue = 2 if div[2] <= div[3] else 3
#     new_state[min_queue] += 1
# elif transition == 2:  # finished service at S2
#     new_state[1] -= 1
#     min_queue = 3 if div[3] <= div[4] else 4
#     new_state[min_queue] += 1
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
# states = list(itertools.product(range(1,N+1), range(N), range(N), range(N), range(N), range(N)))
# avail_states =  [s for s in states if s[0]-s[-1]>0 and  s[0]-s[-1]>1]
# for i in range(N):
#     a = [s for s in states if s[0]-s[-1]==1 and s[0]==i]
#     states = [*a, *states]


#%%
# states = list(itertools.product(range(1,N+1), range(N), range(2*N), range(3*N), range(2*N), range(N)))
# states = [i for i in states if i[0]-i[1]>0]
#%%
start = time.time()
i = 1
wait_times1 = {}
times = {}

total_iterations = len(states)
print(f"total number of iterations needed: {total_iterations}")
parameters= [1,1,1,3,3,3]

for state in states:
    pt = time.time()
    wt, pi3, pi4 = main(state,parameters, wait_times1, N)
    tt = time.time()
    if i%1400==0:
        print(f"i = {i}, time = {tt-pt :.2f} seconds")
        print(f"current state is {state}")
    wait_times1[tuple(state)] = [wt,pi3,pi4]
    times[tuple(state)] = tt-pt
    i += 1

end = time.time()
print(f"Total time:{end-start:.2f} seconds")

#%%
# name = "wait_times" + str(parameters)
# with open('wait_times1.pkl', 'wb') as f:
#     pickle.dump(wait_times1, f)

with open('wait_times1.pkl', 'rb') as f:
    load_wait_times1 = pickle.load(f)

# with open('times.pkl', 'wb') as f:
#     pickle.dump(times, f)

with open('wait_times1.pkl', 'rb') as f:
    load_times = pickle.load(f)
    
    
#%%

main([1, 0, 3, 3, 2, 0], parameters, wait_times1, N)
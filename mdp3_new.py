import heapq
import numpy as np
import time
from numba import njit
import cProfile
import pstats
from dataclasses import dataclass
import itertools
#%%


"""
change heap size and heap priority function, figure out how to use old waiting time values
implement distance to end thing
"""

@dataclass
class State:
    params: np.array #[n1,n2,n3,n4,n5,i] with i=number of customers behind us in Q1
    prob: float #likelihood of state


def main(state,rates, wait_times_dictionary):
    global wait_time, pi3, pi4, mus
    n1, n2, n3, n4, n5, dist_end = state
    lam, mu1, mu2,mu3,mu4, mu5 = rates
    mus = np.array([mu1, mu2, mu3, mu4, mu5, 1])  # mu's plus dummy variable, added for computing div
    rates = [lam, mu1, mu2, mu3, mu4, mu5]

    pi3 = np.zeros(n1 + n3)
    pi4 = np.zeros(n1 + n2 + n4 + 5)

    heap = []
    heapq.heapify(heap)
    root = State(np.array([n1, n2, n3, n4, n5, dist_end], dtype = np.int8), 1.0)
    push(heap, root)

    i = 1
    pt = time.time()
    while np.einsum("i->",pi3)+np.einsum("i->",pi4)<.95:
        
        if heap:
            _, _, best_node = heapq.heappop(heap)
            expand_heap(heap, best_node, rates)
        if i % 10001 == 0:
            tt = time.time()
            diff = 1000 * (tt - pt)
            pt = tt
            avg = np.mean([i[2].prob for i in heap])
            print(f"i={i}, Progress: {sum(pi3) + sum(pi4):.4f}, time = {diff:.2f} ms\naverage probability in heap = {avg:}")        
        i += 1
        
    

    pi_sum = sum(pi3) + sum(pi4)
    pi3 = pi3/pi_sum; pi4 = pi4/pi_sum
    
    state = [n1, n2, n3, n4, n5] 

    return (n1-dist_end - 1)/mu1 + np.dot(pi3,np.array(range(len(pi3))))/mu3 + np.dot(pi4,np.array(range(len(pi4))))/mu4

def expand_heap(heap, elem, rates):
    

    active_transitions = np.array([
        elem.params[0] <= 5 or elem.params[1] <= 5,  # Arrival
        elem.params[0] > 0,  # service at Q1
        elem.params[1] > 0,  # service at Q2
        elem.params[2] > 0,  # service at Q3
        elem.params[3] > 0,  # service at Q4
        elem.params[4] > 0   # service at Q5
    ], dtype =np.bool_)
    asum = np.inner(active_transitions , rates)
    probs = active_transitions * rates / asum
    for i in range(6):
        if (not active_transitions[i]):
            continue

        new_state, min_queue = transition_numba(elem.params, i)

        terminal = new_state[0] == new_state[-1] 

        if terminal:
            if min_queue == 2:
                pi3[new_state[2] - 1] += elem.prob * probs[i]
            elif min_queue == 3:
                pi4[new_state[3] - 1] += elem.prob * probs[i]
            continue
        
        
        push(heap, State(new_state, elem.prob * probs[i]))

@njit
def transition_numba(state, transition):
    min_queue = -1
    new_state = state.copy()
    div = state / mus
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

def push(heap, nd : State):
    heapq.heappush(heap, (1e4/(nd.prob) +( nd.params[0] - nd.params[-1]), id(nd), nd))
    # priority in heap: nr of people in front of us in Q1 (distance to terminal state)
    # divided by likelihood

wt = main([1,0,1,10,0,0],np.array([0.01, 1., 1., 1, 1, 1]), {})
print(wt)
#%%

# profiling the main function
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    wt = main([1,1,1,1,0,0],np.array([0.01, 1., 1., 1, 1, 1]))
    print(wt)

    profiler.disable()

    # display the profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('time').print_stats(2)  # Top 10 results sorted by time


#%%
wait_times = {}
i = 1
start = time.time()
for state in itertools.product(range(1,6), range(5), range(5), range(5), range(5), range(3)):
    if state[0]-state[-1]>0:
        print(wait_times)
        wt = main(state,np.array([0.1,1,0.1,1,0.1,1]), wait_times)
        wait_times[state] = wt
    i += 1
    if i%50 == 0:
        print(i)
    
end = time.time()
print(end - start)
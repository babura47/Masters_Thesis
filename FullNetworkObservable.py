# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:49:18 2024

@author: barba

The first part of the code concerns the policy iteration that results in an 
approximate Nash strategy in the fully observable case, in the full network.
After that follows the simulation code for the network
"""

import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
import pickle
import time
import itertools
from scipy.sparse import coo_array, eye, linalg

plt.rcParams['font.size'] = 30
plt.rcParams['font.family'] = 'DejaVU Sans'
plt.rcParams['font.style'] = 'normal'
#%%

def transition1(state, transition, params, servers_not_full,policy):
    
    """
    makes the transition from one state (n1,n2,n3,n4,n5,i) to another given 
    a transition. transitions that occur because of arrivals follow a policy,
    whereas transitions that occur because of a service at S1 and S2 (where
    the customer must choose a 2nd stage server) follow WJSQ
    
    input:
        state: tuple of the form (n1,n2,n3,n4,n5,i) with i=distance to end of
            queue 1
        transition: int. 0 for arrival, 1 for service at s1, 2 for service at s2 etc.
        params: network parameters
        servers_not_full: specifies which queues are not full and can receive new customers
            if e.g. queue 1 is full and queue 2 not, customers are directed to queue 2
        policy: dictionary which specifies how cusotmers decide between queue 1 and 2
    """
    
    min_queue = -1 #dummy value
    new_state = state.copy()
    mus = params[1:]
    div = state[:-1] / mus  # [n1/mu1, ...., n5/mu5]
    
    if transition == 0:  # arrival
        if servers_not_full[0]==True and servers_not_full[1]==True:
            min_queue = policy[tuple(state[:-1])] 
        elif servers_not_full[0]==False:
            min_queue = 1
        elif servers_not_full[1] == False:
            min_queue = 0
        new_state[min_queue] += 1 
        if min_queue == 0:
            new_state[-1] += 1 # add one to dist to end of queue 1
            
            
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
            min_queue = 3 if div[3] < div[4] else 4
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
    """
    same as transition 1, but we assume the tagged customer is in queue 2, so 
    i = distance to end of queue 2
    """
    
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
            min_queue = 3 if div[3] < div[4] else 4
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
    """
    computes the index of state in the set of states (states1)
    """
    [N1,N2,N3,N4,N5] = Ns
    [n1,n2,n3,n4,n5,i] = state
    idx = n1*N1*N5*N4*N3*N2 + n2*N1*N5*N4*N3 +n3*N1*N5*N4 + n4*N1*N5 +n5*N1 +i
    return idx

def ind2(state, Ns):
    [N1,N2,N3,N4,N5] = Ns
    [n1,n2,n3,n4,n5,i] = state
    idx = n1*N2*N5*N4*N3*(N2+1) + n2*N2*N5*N4*N3 +n3*N2*N5*N4 + n4*N2*(N5) +n5*N2 +i
    return idx

def solve_system1(Ns, params, policy, tol):
    """
    input: 
        Ns = [N1,...,N5], upper bound on kength of queue i (for queue 1 add +1 )
        params = network parameters
        policy = dictionary for joining policy/strategy
        tol = tolerance for iterative solver
    """
    [N1,N2,N3,N4,N5] = Ns
    pt = time.time()
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
            b.append(0) #dummy row for states we do not want to solve for
        else:
            valid_indices.append(it)
            s1_not_full = state[0] <= N1-1 
            s2_not_full = state[1] <= N2-2 
            s3_not_full = state[2] <= N3-2
            s4_not_full = state[3] <= N4-2
            s5_not_full = state[4] <= N5-2
            
            servers_not_full = [s1_not_full, s2_not_full, s3_not_full, s4_not_full, s5_not_full]
        
            # 0/1 vector, i-th entry is 1 if transition is possible
            active_transitions = [
                s1_not_full or s2_not_full,  # Arrival
                state[0] > 0 and (s3_not_full or s4_not_full),  # service at Q1
                state[1] > 0 and (s4_not_full or s5_not_full),  # service at Q2
                state[2] > 0,  # service at Q3
                state[3] > 0,  # service at Q4
                state[4] > 0,  # service at Q5
                ]
          
            asum = np.dot(active_transitions , params) #sum of rates of possible events
            probs = active_transitions * params / asum #vector of transition probabilities
            C = 1/asum
            
            b.append(C) #from the recursive relation in the thesis
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
                    data.append(probs[j]) #adding to the tranition matrix
            active_transitions = np.array(active_transitions)
            
                
        it += 1
            
        if it%(nr_states1//10) ==0:
            progress += 10
            print(f"progress: {progress} %")
    
    
    rows = np.array(row); cols = np.array(col)
    datas = np.array(data)

    A=coo_array((datas, (rows, cols)), shape=(nr_states1,  nr_states1)).tocsr() 
    
    #system is of the form Ax + b = x, or Mx=b with M = I-A

    v = np.transpose(np.array(b));
    I = eye(m = nr_states1, n = nr_states1)
    M = I-A

    # we take only the valid indices
    M = M[valid_indices, :]
    M = M[:, valid_indices]
    v = v[valid_indices]
    tt = time.time()

    print(f"making the matrix took {tt-pt:.2f} seconds")

    wait_times1 = linalg.bicgstab(M,v, atol = tol, rtol=tol)[0]
    
    wait_times1 = wait_times1 - 1/params[1] #take out expected S1 service time for tagged cust
    
    if np.any(wait_times1<-0.1):
        print("negative solutions")
    

    print(f"solving took {time.time()-tt:.2f} seconds")

    
    states1 = [s for s in states1 if s[0]-s[-1]>0]

    wait_times_dict = dict(zip(states1,wait_times1))
    wait_times_dict = {key: value for key, value in wait_times_dict.items() if key[-1] == 0}

    return wait_times_dict, M



def solve_system2(Ns, params, policy,tol):
    """
    this function is the same as solve_system1, only now we assume the tagged customer is in queue 2
    
    """
    pt=time.time()
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
    
    tt = time.time()
    print(f"making the matrix took {tt-pt:.2f} seconds")

    
    wait_times2 = linalg.bicgstab(M,v, atol =tol, rtol=tol)[0]
    
    wait_times2 = wait_times2 - 1/params[2] #take out expected S2 service time for tagged cust

    
    if np.any(wait_times2<-0.1):
        print("negative solutions")
    
    print(f"solving took {time.time()-tt:.2f} seconds")
    
    states2 = [s for s in states2 if s[1]-s[-1]>0]
    wait_times_dict = dict(zip(states2,wait_times2))
    wait_times_dict = {key: value for key, value in wait_times_dict.items() if key[-1] == 0}

    return wait_times_dict, M


#%%

"""
policy iteration
"""

start = time.time()

params = np.array([0.1, 1.5, 1., 1,1,1.])


[N1,N2,N3,N4,N5]= 4*np.ones(5) #upper bound on the queue lengths for which we compute the policy

Ns = [N1,N2,N3,N4,N5] 
states = list(itertools.product(np.arange(N1), np.arange(N2), np.arange(N3), np.arange(N4), np.arange(N5)))

tol = 1e-20

policy= {}
for s in states:
    policy[s] = 0 if (s[0])/params[1] <= (s[1])/params[2] else 1  #WJSQ

for i in range(1,11):
    print(f"i = {i}")
    time1 = time.time()
    wait_times1, _ = solve_system1(Ns,params,policy,tol) 
    time2 = time.time()
    print(f"time = {time2-time1:.2f} seconds")
    wait_times2, _ = solve_system2(Ns,params,policy,tol)
    time3 = time.time()
    print(f"time = {time3-time2:.2f} seconds")

    new_policy= {}
    
    for s in states:
        s1 = list(s).copy(); s1[0]+=1; s1.append(0) #add one to s1[0] because the customer joins queue 1, and append 0 to the end because i=0 then
        s2 = list(s).copy(); s2[1]+=1; s2.append(0)
        if wait_times1[tuple(s1)] <= wait_times2[tuple(s2)]:
            new_policy[s] = 0
        else:
            new_policy[s] = 1
    dis = [s for s in states if new_policy[s]!=policy[s]] #states where the policy changed
    
    print(len(dis))
    
    if len(dis) == 0:
        break
    
    policy = new_policy
    
end = time.time()

print(f"in total this took {(end-start)/60:.2f} minutes")

#%%
#saving the policy and the waiting times 
name = str(params) + str(N1) +"_"+  str(N3) + ".pkl"
with open(name, 'wb') as f:
    pickle.dump(policy, f)
    

name1 = "wt1" + name
with open(name1, 'wb') as f:
    pickle.dump(wait_times1, f)
    
name2 = "wt2" + name
with open(name2, 'wb') as f:
    pickle.dump(wait_times2, f)



#%%
#loading a policy and corresponding waiting times
name = str(params) + str(N1) +"_"+  str(N3) + ".pkl"
with open(name, 'rb') as f:
    policy = pickle.load(f)

name1 = "wt1" + name
with open(name, 'rb') as f:
    wt1 = pickle.load(f)
    
name2 = "wt2" + name
with open(name, 'rb') as f:
    wt2 = pickle.load(f)


#%%
"""
This part of the file contains the code for a simulation of the full network. 

input:
    network parameters
    simulation time
    warm up time (after which we record metrics)
    seed
    policy: 0 for unobservable nash, 1 for wjsq and 2 for approximate Nash
    policy_dict: policy dictionary for Nash
    routing_probs: routing probabilities for unobservable Nash
"""

ARRIVAL_SYSTEM = 0

DEPARTURE_Q1 = 1
DEPARTURE_Q2 = 2
DEPARTURE_Q3 = 3
DEPARTURE_Q4 = 4
DEPARTURE_Q5 = 5


class Event:
    def __init__(self, event_type, event_time, customer_id):
        self.event_type = event_type
        self.event_time = event_time
        self.customer_id = customer_id
    
    def __lt__(self, other):
        return self.event_time < other.event_time

def simulate_network(parameters, simulation_time, warm_up_time, seed, policy, policy_dict, routing_probs ):

    random.seed(seed)
    np.random.seed(seed)    

    [lambda_rate, mu1, mu2, mu3 , mu4, mu5] = parameters
    
    [p,q,r] = routing_probs    
    
    time = 0
    event_queue = []
    customer_id = 0
    total_customers = 0
    
    queue1 = []
    queue2 = []
    queue3 = []
    queue4 = []
    queue5 = []
    
    observability = policy!=0

    server1_busy = False
    server2_busy = False
    server3_busy = False
    server4_busy = False
    server5_busy = False

    departures = [0,0,0,0,0]
    
    total_waiting_time_q1 = 0
    total_waiting_time_q2 = 0
    total_waiting_time_q3 = 0
    total_waiting_time_q4 = 0
    total_waiting_time_q5 = 0

    arrival_times_q1 = {}
    arrival_times_q2 = {}
    arrival_times_q3 = {}
    arrival_times_q4 = {}
    arrival_times_q5 = {}

    service_times_q1 = {}
    service_times_q2 = {}    
    service_times_q3 = {}    
    service_times_q4 = {}    
    service_times_q5 = {}    
    
    decision1 = {} #tracks the first decision of each customer
    decision2 = {} #tracks the second decision of each customer
    
    state_upon_arrival = {}
    state_first_service_done = {}
    
    n_i_max = np.zeros(5)
    
    waiting_times_q1 = {} 
    waiting_times_q2 = {} 
    waiting_times_q3 = {} 
    waiting_times_q4 = {} 
    waiting_times_q5 = {} 
    
    waiting_times = {}
    
    off_policy_customers = [] #stores the id's of customers whos arrival state 
    #is not in the policy dictionary so they use wJSQ instead of Nash
    
    total_wait_time = 0
    
    inter_arrival_time = np.random.exponential(1 / lambda_rate)
    heapq.heappush(event_queue, Event(ARRIVAL_SYSTEM, inter_arrival_time, customer_id))

    customers_disagree = [] #the customers where nash and wjsq disagree    

    while time < simulation_time:

        event = heapq.heappop(event_queue)
        time = event.event_time
        
        if event.event_type == ARRIVAL_SYSTEM:
            """
            EVENT: Arrival of a customer to the system
            
            record the arrival (id and time)
            make a decision of whether to go to queue1 or 2
            
            if chosen server not busy:
                generate service, add the departure event to heap and mark server as busy
                do not record wait  time cause it's 0
            else: 
                add customer to queue
                
            schedule new arrival and add their arrival event to heap
            """
            
            total_customers += 1
            customer_id += 1

            #current state
            n1 = len(queue1)+int(server1_busy)
            n2 = len(queue2)+int(server2_busy)
            n3 = len(queue3)+int(server3_busy)
            n4 = len(queue4)+int(server4_busy)
            n5 = len(queue5)+int(server5_busy)
            
            state_upon_arrival[event.customer_id] = [n1,n2,n3,n4,n5]
            
            #max queue length 
            for k in range(5):
                if state_upon_arrival[event.customer_id][k] > n_i_max[k]:
                    n_i_max[k] = state_upon_arrival[event.customer_id][k]
                
            Q1orQ2unobs = random.choices([1,2], weights=[p,1-p])[0]
            
            if (n1,n2,n3,n4,n5) in policy_dict.keys():
                Q1orQ2nash = policy_dict[(n1,n2,n3,n4,n5)] + 1
            else:
                Q1orQ2nash= 2 if n1/mu1>n2/mu2 else 1  #join shortest queue, if ties go to 1
                off_policy_customers.append(event.customer_id)
                   

            Q1orQ2wjsq = 2 if n1/mu1>n2/mu2 else 1  #join shortest queue, if ties go to 1
           
            #make a decision based on policy we follow
            if policy==0:
                Q1orQ2 = Q1orQ2unobs
            elif policy == 1:
                Q1orQ2 = Q1orQ2wjsq
            elif policy ==2:
                Q1orQ2 = Q1orQ2nash
           
            decision1[event.customer_id] = Q1orQ2 
        
            if Q1orQ2nash != Q1orQ2:
                customers_disagree.append(event.customer_id)
            
            if Q1orQ2 == 1:
                arrival_times_q1[event.customer_id] = time #customer arrives to queue 1
                
                if not server1_busy: # serve customer
                    service_time = np.random.exponential(1/mu1)
                    service_times_q1[event.customer_id] = service_time
                    heapq.heappush(event_queue, Event(DEPARTURE_Q1,time+service_time, event.customer_id))
                    server1_busy = True
                else: 
                    queue1.append(event.customer_id)
                    
            else: #customer goes to queue 2
                arrival_times_q2[event.customer_id] = time
                if not server2_busy: 
                    service_time = np.random.exponential(1/mu2)
                    service_times_q2[event.customer_id] = service_time
                    heapq.heappush(event_queue, Event(DEPARTURE_Q2,time+service_time, event.customer_id))
                    server2_busy = True
                else: 
                    queue2.append(event.customer_id)
            

            next_arrival_time = time + np.random.exponential(1 / lambda_rate)
            heapq.heappush(event_queue, Event(ARRIVAL_SYSTEM, next_arrival_time, customer_id))
        
        elif event.event_type == DEPARTURE_Q1:
            """
            EVENT: Departure from Queue 1 + make a choice of which queue to go next
            
            if queue1 not empty:
                get new customer from the queue, add their departure to event list
                
            """

                
            waiting_time_q1 = time - (arrival_times_q1[event.customer_id] + service_times_q1[event.customer_id] )
            waiting_times_q1[event.customer_id] = waiting_time_q1
            waiting_times[event.customer_id] = waiting_time_q1
            if time>warm_up_time:
                total_waiting_time_q1 += waiting_time_q1
                departures[0] += 1
                total_wait_time += waiting_time_q1
                
            if queue1:
                # Serve the next customer in Queue 1
                next_customer = queue1.pop(0)
                service_time = np.random.exponential(1 / mu1)
                service_times_q1[next_customer] = service_time
                heapq.heappush(event_queue, Event(DEPARTURE_Q1, time + service_time, next_customer))      
            else:
                server1_busy = False  # No one left in Queue 1
            
            
            #current state
            n3 = len(queue3)+int(server3_busy)
            n4 = len(queue4)+int(server4_busy)
            n5 = len(queue5)+int(server5_busy)
            
            state_first_service_done[event.customer_id] = [n3,n4,n5]

            """
            make a choice here between queue 3 and 4
            """
            if observability == False:
                Q3orQ4 = random.choices([3,4], weights=[q,1-q])[0]
            else: 
                # if n3/mu3 != n4/mu4:
                Q3orQ4 = 3 if n3/mu3 <= n4/mu4 else 4
                
    
            decision2[event.customer_id] = Q3orQ4
            
            if Q3orQ4 == 3:
                arrival_times_q3[event.customer_id] = time
                
                if not server3_busy:
                    service_time = np.random.exponential(1 / mu3)
                    service_times_q3[event.customer_id] = service_time
                    heapq.heappush(event_queue, Event(DEPARTURE_Q3, time + service_time, event.customer_id))
                    server3_busy = True
                else:
                    queue3.append(event.customer_id)
                    
            else: 
                arrival_times_q4[event.customer_id] = time
            
                if not server4_busy:
                    service_time = np.random.exponential(1 / mu4)
                    service_times_q4[event.customer_id] = service_time
                    heapq.heappush(event_queue, Event(DEPARTURE_Q4, time + service_time, event.customer_id))
                    server4_busy = True
                else:
                    queue4.append(event.customer_id)
                           
                    
        elif event.event_type == DEPARTURE_Q2:
     
            waiting_time_q2 = time - (arrival_times_q2[event.customer_id] + service_times_q2[event.customer_id]) 
            waiting_times_q2[event.customer_id] = waiting_time_q2
            waiting_times[event.customer_id] = waiting_time_q2

            if time>warm_up_time:
                departures[1] += 1
                total_waiting_time_q2 += waiting_time_q2
                total_wait_time += waiting_time_q2

            
            
            if queue2:
                next_customer = queue2.pop(0)
                service_time = np.random.exponential(1 / mu2)
                service_times_q2[next_customer] = service_time
                heapq.heappush(event_queue, Event(DEPARTURE_Q2, time + service_time, next_customer))
            else:
                server2_busy = False  # No one left in Queue 2, server idles
           
            n3 = len(queue3)+int(server3_busy)
            n4 = len(queue4) + int(server4_busy)
            n5 = len(queue5) + int(server5_busy)
            state_first_service_done[event.customer_id] = [n3,n4,n5]
           
            
            """
            make a choice here between queue 4 and 5
            """
            if observability == False:
                Q4orQ5 = random.choices([4,5], weights=[r,1-r])[0]
            else: 
                Q4orQ5 = 5 if n4/mu4 >= n5/mu5 else 4

            decision2[event.customer_id] = Q4orQ5
            
            if Q4orQ5 == 4:
                arrival_times_q4[event.customer_id] = time
                
                if not server4_busy:
                    service_time = np.random.exponential(1 / mu4)
                    service_times_q4[event.customer_id] = service_time
                    heapq.heappush(event_queue, Event(DEPARTURE_Q4, time + service_time, event.customer_id))
                    server4_busy = True
                else:
                    queue4.append(event.customer_id) 
            else: 
                arrival_times_q5[event.customer_id] = time
            
                if not server5_busy:
                    service_time = np.random.exponential(1 / mu5)
                    service_times_q5[event.customer_id] = service_time
                    heapq.heappush(event_queue, Event(DEPARTURE_Q5, time + service_time, event.customer_id))
                    server5_busy = True
                else:
                    queue5.append(event.customer_id)


        elif event.event_type == DEPARTURE_Q3:

            
            waiting_time_q3 = time - (arrival_times_q3[event.customer_id] + service_times_q3[event.customer_id])
            waiting_times_q3[event.customer_id] = waiting_time_q3
            waiting_times[event.customer_id] += waiting_time_q3

            if time>warm_up_time:
                total_waiting_time_q3 += waiting_time_q3
                departures[2] += 1
                total_wait_time += waiting_time_q3


            if queue3:
                next_customer = queue3.pop(0)
                service_time = np.random.exponential(1 / mu3)
                service_times_q3[next_customer] = service_time

                heapq.heappush(event_queue, Event(DEPARTURE_Q3, time + service_time, next_customer))
            else:
                server3_busy = False  
                
                
        elif event.event_type == DEPARTURE_Q4:
        
            waiting_time_q4 = time - (arrival_times_q4[event.customer_id] + service_times_q4[event.customer_id]) 
            waiting_times_q4[event.customer_id] = waiting_time_q4
            waiting_times[event.customer_id] += waiting_time_q4

            if time>warm_up_time:
                departures[3] += 1
                total_waiting_time_q4 += waiting_time_q4
                total_wait_time += waiting_time_q4
                
            if queue4:
                next_customer = queue4.pop(0)
                service_time = np.random.exponential(1 / mu4)
                service_times_q4[next_customer] = service_time

                heapq.heappush(event_queue, Event(DEPARTURE_Q4, time + service_time, next_customer))
            else:
                server4_busy = False  
                
        elif event.event_type == DEPARTURE_Q5:
        
            waiting_time_q5 = time - (arrival_times_q5[event.customer_id] + service_times_q5[event.customer_id])
            waiting_times_q5[event.customer_id] = waiting_time_q5
            waiting_times[event.customer_id] += waiting_time_q5

            if time>warm_up_time:
                departures[4] += 1
                total_waiting_time_q5 += waiting_time_q5
                total_wait_time += waiting_time_q5
                
            if queue5:
                next_customer = queue5.pop(0)
                service_time = np.random.exponential(1 / mu5)
                service_times_q5[next_customer] = service_time

                heapq.heappush(event_queue, Event(DEPARTURE_Q5, time + service_time, next_customer))
            else:
                server5_busy = False
        
    avg_waiting_time_q1 = total_waiting_time_q1 / departures[0] if departures[0]> 0 else 0 
    avg_waiting_time_q2 = total_waiting_time_q2 / departures[1] if departures[1] > 0 else 0
    avg_waiting_time_q3 = total_waiting_time_q3 / departures[2] if departures[2] > 0 else 0
    avg_waiting_time_q4 = total_waiting_time_q4 / departures[3] if departures[3] > 0 else 0
    avg_waiting_time_q5 = total_waiting_time_q5 / departures[4] if departures[4] > 0 else 0
    
    waiting_times_q4 = dict(sorted(waiting_times_q4.items())) #unsorted because of overtaking
    avg_wait_time = total_wait_time/np.sum(departures[2:])
    
    
    
    return {
        'total_customers': total_customers,
        'departures' : departures, 
        'avg_waiting_time_q1': avg_waiting_time_q1,
        'avg_waiting_time_q2': avg_waiting_time_q2,
        'avg_waiting_time_q3': avg_waiting_time_q3,
        'avg_waiting_time_q4': avg_waiting_time_q4,
        'avg_waiting_time_q5': avg_waiting_time_q5,
        'state_upon_arrival': state_upon_arrival,
        'decision1': decision1,
        'decision2': decision2,
        'waiting_times_q1': waiting_times_q1,
        'waiting_times_q2': waiting_times_q2,
        'waiting_times_q3': waiting_times_q3,
        'waiting_times_q4': waiting_times_q4,
        'waiting_times_q5': waiting_times_q5,
        "n_i_max": n_i_max,
        "avg_wait_time": avg_wait_time,
        "off_policy_customers": off_policy_customers,
        "waiting_times": waiting_times,
        "disagree": customers_disagree
    }


params = np.array([1.6,1.,1.,1.,1.,1.])
simulation_time = 50000
warm_up_time = 1000
# Run the simulation
seed = 10
policy = {} #empty dictionary for Nash -> strategy is wjsq
results = simulate_network(params,  simulation_time,warm_up_time, 10,  2, policy, [0.5,2/3,1/3])

print(results["avg_wait_time"])
print(results["n_i_max"])



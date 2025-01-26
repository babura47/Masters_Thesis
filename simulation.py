# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:49:18 2024

@author: barba
"""

import numpy as np
import heapq
import random
import matplotlib.pyplot as plt

#%%


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

def simulate_network(parameters, simulation_time, policy,seed ):
    
    

    random.seed(seed)
    np.random.seed(seed)    

    [lambda_rate, mu1, mu2, mu3 , mu4, mu5] = parameters
    
    if len(policy)  == 3:
        observability = False
        [p,q,r] = policy
    else: 
        observability = True
    
    time = 0
    event_queue = []
    customer_id = 0
    total_customers = 0
    
    queue1 = []
    queue2 = []
    queue3 = []
    queue4 = []
    queue5 = []

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
    
    off_policy_customers = [] #stores the id's of customers whos arrival state is not in the policy dictionary, so they use wJSQ instead of Nash
    
    total_wait_time = 0
    
    inter_arrival_time = np.random.exponential(1 / lambda_rate)
    heapq.heappush(event_queue, Event(ARRIVAL_SYSTEM, inter_arrival_time, customer_id))
    
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
            
            for k in range(5):
                if state_upon_arrival[event.customer_id][k] > n_i_max[k]:
                    n_i_max[k] = state_upon_arrival[event.customer_id][k]

            if observability == False: 
                Q1orQ2 = random.choices([1,2], weights=[p,1-p])[0] 
            else:
                if type(policy)== dict:
                    
                    if (n1,n2,n3,n4,n5) in policy.keys():
                        Q1orQ2 = policy[(n1,n2,n3,n4,n5)] + 1
                    else:
                        Q1orQ2 = 2 if n1/mu1>=n2/mu2 else 1  #join shortest queue
                        off_policy_customers.append(event.customer_id)
                else:
                    # if n1/mu1 != n2/mu2:
                    Q1orQ2 = 2 if n1/mu1>=n2/mu2 else 1  #join shortest queue
                    # else:
                    #     Q1orQ2 = random.choices([1,2], weights=[0.5, 0.5])[0]
                    
            decision1[event.customer_id] = Q1orQ2 

            
            if Q1orQ2 == 1:
                arrival_times_q1[event.customer_id] = time
                
                if not server1_busy: 
                    service_time = np.random.exponential(1/mu1)
                    service_times_q1[event.customer_id] = service_time
                    heapq.heappush(event_queue, Event(DEPARTURE_Q1,time+service_time, event.customer_id))
                    server1_busy = True
                else: 
                    queue1.append(event.customer_id)
                    
            else:
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
                Q3orQ4 = np.argmin([n3/mu3,n4/mu4]) + 3
                # else:
                #     Q3orQ4 = random.choices([3,4], weights=[0.5,0.5])[0]
                
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
     
            waiting_time_q2 = time - (arrival_times_q2[event.customer_id] + service_times_q2[event.customer_id]) # add minus  service time of customer
            waiting_times_q2[event.customer_id] = waiting_time_q2

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
                # if n4/mu4 != n5/mu5:
                Q4orQ5 = np.argmin([n4/mu4,n5/mu5]) + 4
                # else: 
                #     Q4orQ5 = random.choices([4,5], weights=[0.5, 0.5])[0]
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
        "off_policy_customers": off_policy_customers
    }


params = [1.6,1.,1.,1,1,1]
simulation_time = 500000
warm_up_time = 10000
# Run the simulation
results = simulate_network(params,  simulation_time,[0.5,2/3,1/3], seed=6)

print(results["avg_wait_time"])
print(results["n_i_max"])

#%%



#%%

""" this bit compares the average waiting times if we use policy and if we use wjsq""" 
params=np.array([1., 1., 1., 3., 3., 3.])

simulation_time = 500000
warm_up_time = 10000
nr_sim = 5

avg_waits =[]
avg_waits_wjsq =[]
avg_waits_unobs =[]


N1=10; N3=6
name = str(params) + str(N1) +"_"+  str(N3) +"old"+  ".pkl"
with open(name, 'rb') as f:
    policy = pickle.load(f)

for i in range(nr_sim):
    results = simulate_network(params,  simulation_time,policy, seed=i)
    avg_waits.append(results["avg_wait_time"])
    #print(results["n_i_max"]) 
    
for i in range(nr_sim):
    results = simulate_network(params,  simulation_time,[], seed=i)
    avg_waits_wjsq.append(results["avg_wait_time"])
    #print(results["n_i_max"]) 

for i in range(nr_sim):
    results = simulate_network(params,  simulation_time,[0.5, 2/3, 1/3], seed=i)
    avg_waits_unobs.append(results["avg_wait_time"])
    #print(results["n_i_max"]) 


#print([round(i,3) for i in np.mean(avg_waits, axis= 0)])
avg_waits = np.array(avg_waits)
mean1=np.mean(avg_waits)

sd1=np.sqrt(np.var(avg_waits))
se1=2.086*sd1/np.sqrt(nr_sim)
print("Nash CI is: ", [round(mean1-se1,3), round(mean1+se1,3)])



#print([round(i,3) for i in np.mean(avg_waits, axis= 0)])
avg_waits_wjsq = np.array(avg_waits_wjsq)
mean1=np.mean(avg_waits_wjsq)
sd1=np.sqrt(np.var(avg_waits_wjsq))
se1=2.086*sd1/np.sqrt(nr_sim)
print("wjsq CI is: ", [round(mean1-se1,3), round(mean1+se1,3)])

#print([round(i,3) for i in np.mean(avg_waits, axis= 0)])
avg_waits_unobs = np.array(avg_waits_unobs)
mean1=np.mean(avg_waits_unobs)
sd1=np.sqrt(np.var(avg_waits_unobs))
se1=2.086*sd1/np.sqrt(nr_sim)
print("Unobs CI is: ", [round(mean1-se1,3), round(mean1+se1,3)])

#%%
params=[0.1, 1., 1., 1, 1, 1]

tot_wait = 0
j = 0
val = [0,1,1,1,0]
simulation_time = 100000

for l in range(1):
    results = simulate_network(params,  simulation_time, policy, seed=l)

    state_upon_arrival = results['state_upon_arrival'] 
    decision1 = results['decision1'] 
    decision2 = results['decision2']
    wait_times_q1 = results['waiting_times_q1']
    wait_times_q2 = results['waiting_times_q2']
    wait_times_q3 = results['waiting_times_q3']
    wait_times_q4 = results['waiting_times_q4'] 
    wait_times_q5 = results['waiting_times_q5']
    all_cust = [*list(wait_times_q3.keys()),*list(wait_times_q4.keys()), *list(wait_times_q5.keys())]
    
    customers = [k for k,v in state_upon_arrival.items() if v==val and k in all_cust]
    wait_times = [wait_times_q1, wait_times_q2, wait_times_q3, wait_times_q4, wait_times_q5]

    
    print(len(customers))
    for i in range(len(customers)):
        dec1 = decision1[customers[i]]
        dec2 = decision2[customers[i]]
        wt =  wait_times[dec1-1][customers[i]] + wait_times[dec2-1][customers[i]]
        if dec1==1:
            tot_wait += wt
            j+= 1


print(tot_wait/j)
    



#%%

# obtaining all customers that are done with queueing at stage 2. make a list of all customers done at first stage queueing, then check
# for each of them if they're in wait_times_q3 or 4 or 5. 

# customers




# cust_1 = [k for k,v in decision1.items() if v==1] #all custommers who went to server 2
# cust_2 = [k for k,v in decision1.items() if v==2] #all custommers who went to server 2
# cust_3 = [k for k,v in decision2.items() if v==3] 
# cust_4 = [k for k,v in decision2.items() if v==4]
# cust_5 = [k for k,v in decision2.items() if v==5]
  
# cust_13 = [k for k in cust_1 if k in cust_3]
# cust_14 = [k for k in cust_1 if k in cust_4]
# cust_24 = [k for k in cust_2 if k in cust_4]
# cust_25 = [k for k in cust_2 if k in cust_5]


# itemgetter(*cust_13)(wait_times_q1)

# n1max = np.max([v[0] for k,v in state_upon_arrival.items()])
# n2max = np.max([v[1] for k,v in state_upon_arrival.items()])
# n3max = np.max([v[2] for k,v in state_upon_arrival.items()])
# n4max = np.max([v[3] for k,v in state_upon_arrival.items()])
# n5max = np.max([v[4] for k,v in state_upon_arrival.items()])

best_decision= {}

for j in range(49,50):
    val = [0,0,1,1,0]
    print(val)
    customers = [k for k,v in state_upon_arrival.items() if v==val and k in all_cust]
    print(customers)
    wait_times = [wait_times_q1, wait_times_q2, wait_times_q3, wait_times_q4, wait_times_q5]
    dec1 = decision1[customers[0]]
    dec2 = decision2[customers[0]]
    wait_time_min = wait_times[dec1-1][customers[0]] + wait_times[dec2-1][customers[0]]
    wait_time_min_customer = 0
    for i in customers:
        dec1 = decision1[i]
        dec2 = decision2[i]
        wait_time = wait_times[dec1-1][i] + wait_times[dec2-1][i]
        if wait_time<=wait_time_min: 
            wait_time_min = wait_time
            wait_time_min_customer = i
    
    best_decision[j] = [val, dec1, dec2] 
    #print( state_upon_arrival[wait_time_min_customer], decision1[wait_time_min_customer], decision2[wait_time_min_customer])
    
    
    



#%%
val = [0,0,1,1,0]
customers = [k for k,v in state_upon_arrival.items() if v==val and k in all_cust]
wait_times = [wait_times_q1, wait_times_q2, wait_times_q3, wait_times_q4, wait_times_q5]

tot_wait = 0; j=0
print(len(customers))
for i in range(len(customers)):
    dec1 = decision1[customers[i]]
    dec2 = decision2[customers[i]]
    wt =  wait_times[dec1-1][customers[i]] + wait_times[dec2-1][customers[i]]
    #print([wait_times[dec1-1][customers[i]], wait_times[dec2-1][customers[i]]] )
    if dec1==1:
        tot_wait += wt
        j+= 1

print(tot_wait/j)





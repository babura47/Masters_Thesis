# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:49:18 2024

@author: barba
"""

import numpy as np
import heapq
import random
from operator import itemgetter
import SimplifiedNetwork as sn
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 30
plt.rcParams['font.family'] = 'DejaVU Sans'
plt.rcParams['font.style'] = 'normal'

ARRIVAL_SYSTEM = 0

DEPARTURE_Q1 = 1
DEPARTURE_Q2 = 2
DEPARTURE_Q3 = 3
DEPARTURE_Q4 = 4


class Event:
    def __init__(self, event_type, event_time, customer_id):
        self.event_type = event_type
        self.event_time = event_time
        self.customer_id = customer_id
    
    def __lt__(self, other):
        return self.event_time < other.event_time

def simulate_network(parameters, simulation_time, seed):
    
    random.seed(seed)
    np.random.seed(seed)
    
    [lambda_rate, mu1, mu2, mu3 , mu4] = parameters
        
    time = 0
    event_queue = []
    customer_id = 0
    total_customers = 0
    
    queue1 = []
    queue2 = []
    queue3 = []
    queue4 = []
    
    server1_busy = False
    server2_busy = False
    server3_busy = False
    server4_busy = False

    departures = [0,0,0,0,0]
    
    A, B = sn.expected_waiting_time(parameters, [100,100,100,100])

    
    total_waiting_time_q1 = 0
    total_waiting_time_q2 = 0
    total_waiting_time_q3 = 0
    total_waiting_time_q4 = 0

    arrival_times_q1 = {}
    arrival_times_q2 = {}
    arrival_times_q3 = {}
    arrival_times_q4 = {}

    service_times_q1 = {}
    service_times_q2 = {}    
    service_times_q3 = {}    
    service_times_q4 = {}    
    
    decision = {} #tracks the first decision of each customer
    
    state_upon_arrival = {}
    
    waiting_times_q1 = {} 
    waiting_times_q2 = {} 
    waiting_times_q3 = {} 
    waiting_times_q4 = {} 
    
    max_queue_length_1 = 0
    max_queue_length_2 = 0
    max_queue_length_3 = 0
    max_queue_length_4 = 0
    
    total_wait_time = {} # total wait time of each customer
    
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
            
            state_upon_arrival[event.customer_id] = [n1,n2,n3,n4]
            
            """
            unobservable strategy
            """
            Q1orQ2 =  random.choices([1,2], weights=[0.5, 0.5])[0]
            
            
            
            # """
            # naive strategy
            # """
            # if n1/mu1 != n2/mu2:
            #     Q1orQ2 = np.argmin([n1/mu1,n2/mu2]) + 1 #join shortest queue
            # else:
            #     Q1orQ2 = random.choices([1,2], weights=[0.5, 0.5])[0]
                
            
            # """
            # Nash strategy
            # """
            # EW13 = A[n1,n3]
            # EW24 = B[n2,n4]
            
            # if EW13 == EW24:
            #     Q1orQ2 = random.choices([1,2], weights=[0.5, 0.5])[0]
            # else: 
            #     Q1orQ2 = np.argmin([EW13,EW24]) + 1
                   
                
            # """
            # strategy that discounts starvation
            # """
            # EW13 = sn.wait_times_old([mu1,mu3],[n1,n3])
            # EW24 = sn.wait_times_old([mu2,mu4],[n2,n4])
            # if EW13 == EW24:
            #     Q1orQ2 = random.choices([1,2], weights=[0.5, 0.5])[0]
            # else: 
            #     Q1orQ2 = np.argmin([EW13,EW24]) + 1    
                
                
            decision[event.customer_id] = Q1orQ2 

            
            if Q1orQ2 == 1:
                arrival_times_q1[event.customer_id] = time
                
                if not server1_busy: 
                    service_time = np.random.exponential(1/mu1)
                    service_times_q1[event.customer_id] = service_time
                    heapq.heappush(event_queue, Event(DEPARTURE_Q1,time+service_time, event.customer_id))
                    server1_busy = True
                else: 
                    queue1.append(event.customer_id)
                    if len(queue1) > max_queue_length_1:
                        max_queue_length_1 = len(queue1)
                    
            else:
                arrival_times_q2[event.customer_id] = time
                if not server2_busy: 
                    service_time = np.random.exponential(1/mu2)
                    service_times_q2[event.customer_id] = service_time
                    heapq.heappush(event_queue, Event(DEPARTURE_Q2,time+service_time, event.customer_id))
                    server2_busy = True
                else: 
                    queue2.append(event.customer_id)
                    if len(queue2) > max_queue_length_2:
                        max_queue_length_2 = len(queue2)            

            
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
                
            if queue1:
                # Serve the next customer in Queue 1
                next_customer = queue1.pop(0)
                service_time = np.random.exponential(1 / mu1)
                service_times_q1[next_customer] = service_time
                heapq.heappush(event_queue, Event(DEPARTURE_Q1, time + service_time, next_customer))      
            else:
                server1_busy = False  # No one left in Queue 1
            
            
            
            arrival_times_q3[event.customer_id] = time
            if not server3_busy:
                service_time = np.random.exponential(1 / mu3)
                service_times_q3[event.customer_id] = service_time
                heapq.heappush(event_queue, Event(DEPARTURE_Q3, time + service_time, event.customer_id))
                server3_busy = True
            else:
                queue3.append(event.customer_id)
                if len(queue3) > max_queue_length_3:
                    max_queue_length_3 = len(queue3)
   
                           
                    
        elif event.event_type == DEPARTURE_Q2:
     
            waiting_time_q2 = time -( arrival_times_q2[event.customer_id] + service_times_q2[event.customer_id]) # add minus  service time of customer
            waiting_times_q2[event.customer_id] = waiting_time_q2

            if time>warm_up_time:
                departures[1] += 1
                total_waiting_time_q2 += waiting_time_q2
            
            
            
            if queue2:
                next_customer = queue2.pop(0)
                service_time = np.random.exponential(1 / mu2)
                service_times_q2[next_customer] = service_time
                heapq.heappush(event_queue, Event(DEPARTURE_Q2, time + service_time, next_customer))
            else:
                server2_busy = False  # No one left in Queue 2, server idles
           

 
            arrival_times_q4[event.customer_id] = time
            
            if not server4_busy:
                service_time = np.random.exponential(1 / mu4)
                service_times_q4[event.customer_id] = service_time
                heapq.heappush(event_queue, Event(DEPARTURE_Q4, time + service_time, event.customer_id))
                server4_busy = True
            else:
                queue4.append(event.customer_id) 
                if len(queue4) > max_queue_length_4:
                    max_queue_length_4 = len(queue4)        


        elif event.event_type == DEPARTURE_Q3:

            
            waiting_time_q3 = time - (arrival_times_q3[event.customer_id] + service_times_q3[event.customer_id])
            waiting_times_q3[event.customer_id] = waiting_time_q3
            
            if time>warm_up_time:
                total_waiting_time_q3 += waiting_time_q3
                departures[2] += 1

            if queue3:
                next_customer = queue3.pop(0)
                service_time = np.random.exponential(1 / mu3)
                service_times_q3[next_customer] = service_time

                heapq.heappush(event_queue, Event(DEPARTURE_Q3, time + service_time, next_customer))
            else:
                server3_busy = False  
                
            if decision[event.customer_id] ==1:
                total_wait_time[event.customer_id] = waiting_time_q3 + waiting_times_q1[event.customer_id]
            else:
                total_wait_time[event.customer_id] = waiting_time_q3 + waiting_times_q2[event.customer_id]
                
        elif event.event_type == DEPARTURE_Q4:
        
            waiting_time_q4 = time - (arrival_times_q4[event.customer_id] + service_times_q4[event.customer_id]) 
            waiting_times_q4[event.customer_id] = waiting_time_q4
            if time>warm_up_time:
                departures[3] += 1
                total_waiting_time_q4 += waiting_time_q4
                
            if queue4:
                next_customer = queue4.pop(0)
                service_time = np.random.exponential(1 / mu4)
                service_times_q4[next_customer] = service_time

                heapq.heappush(event_queue, Event(DEPARTURE_Q4, time + service_time, next_customer))
            else:
                server4_busy = False  
                
            if decision[event.customer_id] ==1:
                total_wait_time[event.customer_id] = waiting_time_q4 + waiting_times_q1[event.customer_id]
            else:
                total_wait_time[event.customer_id] = waiting_time_q4 + waiting_times_q2[event.customer_id]
                
                
      
        
    avg_waiting_time_q1 = total_waiting_time_q1 / departures[0] if departures[0]> 0 else 0 
    avg_waiting_time_q2 = total_waiting_time_q2 / departures[1] if departures[1] > 0 else 0
    avg_waiting_time_q3 = total_waiting_time_q3 / departures[2] if departures[2] > 0 else 0
    avg_waiting_time_q4 = total_waiting_time_q4 / departures[3] if departures[3] > 0 else 0
    
    dict(sorted(total_wait_time.items()))
    
    return {
        'total_customers': total_customers,
        'departures' : departures, 
        'avg_waiting_time_q1': avg_waiting_time_q1,
        'avg_waiting_time_q2': avg_waiting_time_q2,
        'avg_waiting_time_q3': avg_waiting_time_q3,
        'avg_waiting_time_q4': avg_waiting_time_q4,
        'state_upon_arrival': state_upon_arrival,
        'decision': decision,
        'waiting_times_q1': waiting_times_q1,
        'waiting_times_q2': waiting_times_q2,
        'waiting_times_q3': waiting_times_q3,
        'waiting_times_q4': waiting_times_q4,
        'max_queue_1': max_queue_length_1,
        'max_queue_2': max_queue_length_2,
        'max_queue_3': max_queue_length_3,
        'max_queue_4': max_queue_length_4,
        "arrival_times_q1": arrival_times_q1,
        "arrival_times_q2": arrival_times_q2,
        "total_wait_time": total_wait_time
    }


# Parameters
lam = 1.8
mu1 = 1.   
mu2 = 1. 
mu3 = 1.
mu4 = 1.

params=[lam,mu1,mu2,mu3, mu4]

simulation_time = 50000
warm_up_time = 5000

avg_waits = np.zeros([20,2])

for i in range(20):
    # Run the simulation
    print(i)
    results = simulate_network(params,  simulation_time, seed=i)
    avg_waits[i,0] = results['avg_waiting_time_q1'] + results['avg_waiting_time_q3']
    avg_waits[i,1] = results['avg_waiting_time_q2'] + results['avg_waiting_time_q4']

#%%

# %%

from scipy.stats import expon

arr_times_q1 = np.array(list(results["arrival_times_q1"].values()))
interarr_q1 = arr_times_q1[1:] - arr_times_q1[:-1]

plt.figure()
plt.hist(interarr_q1, weights=np.ones_like(interarr_q1) / len(interarr_q1), bins=40, density=True, alpha=0.6, color='skyblue')

mean_interarrival = np.mean(interarr_q1)
x = np.linspace(0, np.max(interarr_q1), 100)
exp_density = expon.pdf(x, scale=mean_interarrival)

plt.plot(x, exp_density, 'r-', lw=2, label=f"exponential density, mean ={mean_interarrival:.2f}")
plt.title('Interarrival times to queue 1 in the simplified network\n$\mu_1=\mu_2=\mu_3=\mu_4=1,\lambda=1.8$')
plt.xlabel("Interarrival Time")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#%%
arr_times_q2 = np.array(list(results["arrival_times_q2"].values()))
interarr_q2 = arr_times_q2[1:] - arr_times_q2[:-1]

plt.figure()
plt.hist(interarr_q2, weights=np.ones_like(interarr_q2) / len(interarr_q2), bins=40, density=True, alpha=0.6, color='skyblue')

mean_interarrival = np.mean(interarr_q2)
x = np.linspace(0, np.max(interarr_q2), 100)
exp_density = expon.pdf(x, scale=mean_interarrival)

plt.plot(x, exp_density, 'r-', lw=2, label=f"exponential density, mean ={mean_interarrival:.2f}")

plt.xlabel("Interarrival Time")
plt.ylabel("Density")
plt.legend()
plt.show()


#%%

# all customers that left the system

max_q_len_1=results['max_queue_1']
max_q_len_2=results['max_queue_2']
max_q_len_3=results['max_queue_3']
max_q_len_4=results['max_queue_4']


state_upon_arrival = results['state_upon_arrival'] 
decision = results['decision'] 
wait_times_q1 = results['waiting_times_q1']
wait_times_q2 = results['waiting_times_q2']
wait_times_q3 = results['waiting_times_q3']
wait_times_q4 = results['waiting_times_q4'] 

all_cust = [*list(wait_times_q3.keys()),*list(wait_times_q4.keys())]

total_wait_time = results['total_wait_time']

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


states = list(state_upon_arrival.values())
for j in range(1,3):
    
    val = list(state_upon_arrival.values())[j]
    print(val)
    customers = [k for k,v in state_upon_arrival.items() if v==val and k in all_cust]
    wait_times = [wait_times_q1, wait_times_q2, wait_times_q3, wait_times_q4]
    dec1 = decision[customers[0]]
    
    for i in customers:
        dec = decision[i]
        wait_time = total_wait_time[i]
        print(dec, wait_time)
    

    #print( state_upon_arrival[wait_time_min_customer], decision1[wait_time_min_customer], decision2[wait_time_min_customer])
    


# %%



from collections import defaultdict

# Original dictionary
dic = {0: 'a', 1: 'b', 2: 'b', 3: 'c', 4: 'a', 5: 'c'}

# Create a defaultdict with dict as the default factory
separated_dicts = defaultdict(dict)

# Separate items based on values
for key, value in dic.items():
    separated_dicts[value][key] = value

# Convert defaultdict to a regular dictionary if desired
separated_dicts = dict(separated_dicts)

# Output the result
for value, sub_dict in separated_dicts.items():
    print(f"Dictionary for value '{value}':", sub_dict)










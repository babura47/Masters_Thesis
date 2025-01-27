# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:04:20 2025

@author: barba
"""


import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
import itertools
import time
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

def simulate_network_short(parameters, sim_time,warm_up_time, seed, decision, policy_dict, init_state):

    random.seed(seed)
    np.random.seed(seed)    

    [lambda_rate, mu1, mu2, mu3 , mu4, mu5] = parameters
        
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
    
    
    waiting_times_q1 = {} 
    waiting_times_q2 = {} 
    waiting_times_q3 = {} 
    waiting_times_q4 = {} 
    waiting_times_q5 = {} 
    
    waiting_times = {}
            
    inter_arrival_time = np.random.exponential(1 / lambda_rate)
    heapq.heappush(event_queue, Event(ARRIVAL_SYSTEM, inter_arrival_time, customer_id))

    tagged_customer_id = -1
    tagged_customer_done = False

    while tagged_customer_done==False and time<sim_time:
        
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
            
            if time> warm_up_time:
                if state_upon_arrival[event.customer_id] == init_state and tagged_customer_id == -1:
                    tagged_customer_id = event.customer_id
        
            if event.customer_id == tagged_customer_id:
                Q1orQ2 = decision
            else:
                if (n1,n2,n3,n4,n5) in policy_dict.keys():
                    Q1orQ2 = policy_dict[(n1,n2,n3,n4,n5)] + 1
                else:
                    Q1orQ2= 2 if n1/mu1>n2/mu2 else 1  #join shortest queue, if ties go to 1                   
                    
                
                
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
            waiting_times[event.customer_id] = waiting_time_q1
          
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
     
            waiting_time_q2 = time - (arrival_times_q2[event.customer_id] + service_times_q2[event.customer_id]) # add minus  service time of customer
            waiting_times_q2[event.customer_id] = waiting_time_q2
            waiting_times[event.customer_id] = waiting_time_q2

            
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
            if event.customer_id == tagged_customer_id:
                tagged_customer_done = True
         
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
            if event.customer_id == tagged_customer_id:
                tagged_customer_done = True
                
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
            if event.customer_id == tagged_customer_id:
                tagged_customer_done = True
            if queue5:
                next_customer = queue5.pop(0)
                service_time = np.random.exponential(1 / mu5)
                service_times_q5[next_customer] = service_time

                heapq.heappush(event_queue, Event(DEPARTURE_Q5, time + service_time, next_customer))
            else:
                server5_busy = False
        
   
    
    waiting_times_q4 = dict(sorted(waiting_times_q4.items())) #unsorted because of overtaking
    
    
    
    return waiting_times[tagged_customer_id]
       
    

nr_sim = 25
params = [1.,1.,1.,1,1,1]
simulation_time = 50000
warm_up_time = 1000
# Run the simulation
seed = 10
decision = 1
wait_time2 =  simulate_network_short(params, sim_time, warm_up_time, i, 2, policy_dict, init_state)


#%%
nr_sim = 50
params = [1.,1.,1.,1,1,1]
warm_up_time = 100
# Run the simulation
seed = 10
decision = 1
policy_dict=policy
sim_time = 500000
for state in itertools.product(range(3),range(3),range(3),range(3),range(3)):
    init_state = list(state)
    wt1 = []
    wt2 = []
    for i in range(nr_sim):
        wait_time1 =  simulate_network_short(params, sim_time, warm_up_time, i, 1, policy_dict, init_state)
        wait_time2 =  simulate_network_short(params, sim_time, warm_up_time, i, 2, policy_dict, init_state)
        wt1.append(wait_time1)
        wt2.append(wait_time2)
    
    if policy[tuple(init_state)] == 0 and np.mean(wt1) <= np.mean(wt2):
        print("good")
    elif policy[tuple(init_state)] == 1 and np.mean(wt1) > np.mean(wt2):
        print("good")
    else:
        print("bad", state)
        
#%%
start = time.time()
nr_sim = 1
warm_up_time = 0
sim_time = 100
init_state = [0, 2, 0, 2, 2]
wt1 = []
wt2 = []
for i in range(nr_sim):
    wait_time1 =  simulate_network_short(params, sim_time, warm_up_time, i, 1, policy_dict, init_state)
    wait_time2 =  simulate_network_short(params, sim_time, warm_up_time, i, 2, policy_dict, init_state)
    wt1.append(wait_time1)
    wt2.append(wait_time2)

if policy[tuple(init_state)] == 0 and np.mean(wt1) <= np.mean(wt2):
    print("good")
elif policy[tuple(init_state)] == 1 and np.mean(wt1) > np.mean(wt2):
    print("good")
else:
    print("bad")
    print(state)
print(time.time()-start)
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:49:18 2024

@author: barba
"""

import numpy as np
import heapq
import random

"""
Add warmup time to simulation
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

def simulate_tandem_mm1_queue(parameters, simulation_time):
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
    
    decision1 = {} #tracks for each customer if they went to queue 1  or queue 2 
    decision2 = {} #tracks for each customer if they went to queue 3  or queue 4
    decision3 = {} #tracks for each customer if they went to queue 4  or queue 5
    
    state_upon_arrival = {}
    
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
            state_upon_arrival[customer_id] = [n1,n2,n3,n4,n5]
            
            Q1orQ2 = random.choices([1,2], weights=[0.5,0.5])[0] 
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
            departures[0] += 1
            
            waiting_time_q1 = time - arrival_times_q1[event.customer_id] - service_times_q1[event.customer_id] 
            total_waiting_time_q1 += waiting_time_q1
            
            if queue1:
                # Serve the next customer in Queue 1
                next_customer = queue1.pop(0)
                service_time = np.random.exponential(1 / mu1)
                service_times_q1[next_customer] = service_time
                heapq.heappush(event_queue, Event(DEPARTURE_Q1, time + service_time, next_customer))
                
            else:
                server1_busy = False  # No one left in Queue 1
            
            
            """
            make a choice here between queue 3 and 4
            """
            branch = random.choices([3,4], weights=[0.5,0.5])[0]
            decision2[event.customer_id] = branch
            
            if branch == 3:
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
            departures[1] += 1

            waiting_time_q2 = time - arrival_times_q2[event.customer_id] - service_times_q2[event.customer_id] # add minus  service time of customer
            total_waiting_time_q2 += waiting_time_q2
            
            if queue2:
                next_customer = queue2.pop(0)
                service_time = np.random.exponential(1 / mu2)
                service_times_q2[next_customer] = service_time
                heapq.heappush(event_queue, Event(DEPARTURE_Q2, time + service_time, next_customer))
            else:
                server2_busy = False  # No one left in Queue 2, server idles
           
            """
            make a choice here between queue 4 and 5
            """
            branch = random.choices([4,5], weights=[0.5,0.5])[0]
            decision3[event.customer_id] = branch
            
            if branch == 4:
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
            departures[2] += 1

            
            waiting_time_q3 = time - arrival_times_q3[event.customer_id] - service_times_q3[event.customer_id] # add minus  service time of customer
            total_waiting_time_q3 += waiting_time_q3
            
            if queue3:
                next_customer = queue3.pop(0)
                service_time = np.random.exponential(1 / mu3)
                service_times_q3[next_customer] = service_time

                heapq.heappush(event_queue, Event(DEPARTURE_Q3, time + service_time, next_customer))
            else:
                server3_busy = False  
                
                
        elif event.event_type == DEPARTURE_Q4:
            departures[3] += 1

            waiting_time_q4 = time - arrival_times_q4[event.customer_id] - service_times_q4[event.customer_id] # add minus  service time of customer
            total_waiting_time_q4 += waiting_time_q4
            
            if queue4:
                next_customer = queue4.pop(0)
                service_time = np.random.exponential(1 / mu4)
                service_times_q4[next_customer] = service_time

                heapq.heappush(event_queue, Event(DEPARTURE_Q4, time + service_time, next_customer))
            else:
                server4_busy = False  
                
        elif event.event_type == DEPARTURE_Q5:
            
            departures[4] += 1

            
            waiting_time_q5 = time - arrival_times_q5[event.customer_id] - service_times_q5[event.customer_id] # add minus  service time of customer
            total_waiting_time_q5 += waiting_time_q5
            
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
        
    return {
        'total_customers': total_customers,
        'departures' : departures, 
        'avg_waiting_time_q1': avg_waiting_time_q1,
        'avg_waiting_time_q2': avg_waiting_time_q2,
        'avg_waiting_time_q3': avg_waiting_time_q3,
        'avg_waiting_time_q4': avg_waiting_time_q4,
        'avg_waiting_time_q5': avg_waiting_time_q5,
        'state_upon_arrival': state_upon_arrival,
        'decisions1': decision1,
        'decisions2': decision2,
        'decisions3': decision3
    }

[p]=[0.5]

# Parameters
lam = 2.0
mu1 = 3.0; lam1= p*lam       
mu2 = 2.; lam2=p*lam       
mu3 = 2.; lam3 = p*p*lam 
mu4=3.; lam4 = 2*(p*(1-p))*lam
mu5=1.; lam5= p*p*lam
params=[lam,mu1,mu2,mu3, mu4,mu5]

simulation_time = 10 

# Run the simulation
results = simulate_tandem_mm1_queue(params,  simulation_time)

# Output results
print(f"Total customers arrived: {results['total_customers']}")
print(f"Total customers departed Queue 1: {results['departures'][0]}")
print(f"Total customers departed Queue 2: {results['departures'][1]}")
print(f"Total customers departed Queue 3: {results['departures'][2]}")
print(f"Total customers departed Queue 4: {results['departures'][3]}")
print(f"Total customers departed Queue 5: {results['departures'][4]}")

print(f"Average waiting time in Queue 1: {results['avg_waiting_time_q1']:.4f}, expected {round(lam1/(mu1*(mu1-lam1)),3)}")
print(f"Average waiting time in Queue 2: {results['avg_waiting_time_q2']:.4f}, expected {round(lam2/(mu2*(mu2-lam2)),3)}")
print(f"Average waiting time in Queue 3: {results['avg_waiting_time_q3']:.4f}, expected {round(lam3/(mu3*(mu3-lam3)),3)}")
print(f"Average waiting time in Queue 4: {results['avg_waiting_time_q4']:.4f}, expected {round(lam4/(mu4*(mu4-lam4)),3)}")
print(f"Average waiting time in Queue 5: {results['avg_waiting_time_q5']:.4f}, expected {round(lam5/(mu5*(mu5-lam5)),3)}")

print()



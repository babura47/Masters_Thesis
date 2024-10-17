# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:49:18 2024

@author: barba
"""

import numpy as np
import heapq

"""
Add warmup time to simulation
"""

# Event types
ARRIVAL_Q1 = 1
DEPARTURE_Q1 = 2
DEPARTURE_Q2 = 3

class Event:
    def __init__(self, event_type, event_time, customer_id):
        self.event_type = event_type
        self.event_time = event_time
        self.customer_id = customer_id
    
    def __lt__(self, other):
        return self.event_time < other.event_time

def simulate_tandem_mm1_queue(lambda_rate, mu1, mu2, simulation_time):
    time = 0
    event_queue = []
    customer_id = 0
    total_customers = 0
    
    # State variables for the queues
    queue1 = []
    queue2 = []
    
    server1_busy = False
    server2_busy = False
    
    # Statistics
    departures_q1 = 0
    departures_q2 = 0
    total_waiting_time_q1 = 0
    total_waiting_time_q2 = 0
    
    # Track arrival times of customers to calculate waiting times
    arrival_times_q1 = {}
    arrival_times_q2 = {}

    # Track service times of customers to calculate waiting times
    service_times_q1 = {}
    service_times_q2 = {}    
    
    
    # First arrival
    inter_arrival_time = np.random.exponential(1 / lambda_rate)
    heapq.heappush(event_queue, Event(ARRIVAL_Q1, inter_arrival_time, customer_id))
    
    while time < simulation_time:
        # Get the next event from the heap
        event = heapq.heappop(event_queue)
        time = event.event_time

        if event.event_type == ARRIVAL_Q1:
            
            """
            EVENT: Arrival to queue 1
            
            record the arrival (id and time)
            if server not busy:
                generate service, add the departure event to heap and mark serever as busy
                do not record wait  time cause it's 0
            else: 
                add customer to queue1
                
            schedule new arrival and add their arrival event to heap
            """
            
            # New arrival at Queue 1
            total_customers += 1
            customer_id += 1
            
            # Record the arrival time to calculate waiting time later
            arrival_times_q1[event.customer_id] = time
            
            if not server1_busy:
                # If server 1 is free, serve immediately
                service_time_1 = np.random.exponential(1 / mu1)
                service_times_q1[event.customer_id] = service_time_1
                heapq.heappush(event_queue, Event(DEPARTURE_Q1, time + service_time_1, event.customer_id))
                server1_busy = True 
            else:
                # Otherwise, wait in queue 1
                queue1.append(event.customer_id)
            
            # Schedule next arrival
            next_arrival_time = time + np.random.exponential(1 / lambda_rate)
            
            
            heapq.heappush(event_queue, Event(ARRIVAL_Q1, next_arrival_time, customer_id))
        
        elif event.event_type == DEPARTURE_Q1:
            """
            EVENT: Departure from Queue 1, arrival at Queue 2
            
            if queue1 not empty:
                get new customer from the queue, add their departure to event list
                
            """
            
            # Departure from Queue 1 
            departures_q1 += 1
            
            if queue1:
                # Serve the next customer in Queue 1
                next_customer = queue1.pop(0)
                service_time_q1 = np.random.exponential(1 / mu1)
                service_times_q1[next_customer] = service_time_q1
                heapq.heappush(event_queue, Event(DEPARTURE_Q1, time + service_time_q1, next_customer))
                
                # Calculate waiting time for Queue 1 (departure time - arrival time)
                waiting_time_q1 = time - arrival_times_q1[next_customer]
                waiting_time_q1 = time - service_times_q1[event.customer_id] - arrival_times_q1[event.customer_id] 
                
                total_waiting_time_q1 += waiting_time_q1
            else:
                server1_busy = False  # No one left in Queue 1
            
            # Record the time of arrival in Queue 2 for calculating waiting time in Queue 2
            arrival_times_q2[event.customer_id] = time
            
            # Arriving in Queue 2
            if not server2_busy:
                # If server 2 is free, serve immediately
                service_time_q2 = np.random.exponential(1 / mu2)
                service_times_q2[event.customer_id] = service_time_q2
                heapq.heappush(event_queue, Event(DEPARTURE_Q2, time + service_time_q2, event.customer_id))
                server2_busy = True
            else:
                # Otherwise, wait in Queue 2
                queue2.append(event.customer_id)
        
        elif event.event_type == DEPARTURE_Q2:
            # Departure from Queue 2 (customer leaves system)
            departures_q2 += 1
            
            # Calculate waiting time for Queue 2 (departure time - arrival time in Queue 2)
            waiting_time_q2 = time - arrival_times_q2[event.customer_id] - service_times_q2[event.customer_id] # add minus  service time of customer
            total_waiting_time_q2 += waiting_time_q2
            
            if queue2:
                # Serve the next customer in Queue 2
                next_customer = queue2.pop(0)
                service_time_q2 = np.random.exponential(1 / mu2)
                service_times_q2[next_customer] = service_time_q2

                heapq.heappush(event_queue, Event(DEPARTURE_Q2, time + service_time_q2, next_customer))
            else:
                server2_busy = False  # No one left in Queue 2
        
    # Calculate the average waiting time in each queue
    avg_waiting_time_q1 = total_waiting_time_q1 / total_customers if total_customers > 0 else 0 
    avg_waiting_time_q2 = total_waiting_time_q2 / departures_q2 if departures_q2 > 0 else 0
    
    return {
        'total_customers': total_customers,
        'departures_q2': departures_q2,
        'avg_waiting_time_q1': avg_waiting_time_q1,
        'avg_waiting_time_q2': avg_waiting_time_q2
    }

# Parameters
lam = 2.0
mu1 = 3.0         
mu2 = 2.5         
simulation_time = 10000  


# Run the simulation
results = simulate_tandem_mm1_queue(lam, mu1, mu2, simulation_time)

# Output results
print(f"Total customers arrived: {results['total_customers']}")
print(f"Total customers departed Queue 2: {results['departures_q2']}")
print(f"Average waiting time in Queue 1: {results['avg_waiting_time_q1']:.4f}")
print(f"Average waiting time in Queue 2: {results['avg_waiting_time_q2']:.4f}")

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:49:43 2024

@author: barba

The first part of the file contains functions which are used to compute the (approximate) expected waiting 
times given a tandem {S1,S3} in three different ways: exact, using the deterministic
approximation and using the "no starvation" approximation. These waiting times are
used when deriving the Nash equilibrium for the simplified network, which is made
up of two such tandems.
The second part contains the code for the simulation of the simplified network
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
import random
import matplotlib


plt.rcParams['font.size'] = 30
plt.rcParams['font.family'] = 'DejaVU Sans'
plt.rcParams['font.style'] = 'normal'
#%%

def tandem_wait_times(mu1,mu3, n1max,n3max):
    """
    returns a matrix A of size n1max by n3max+n1max such that A[i,j] = E(clearing time, n1=i, n2=j). 
    ni is the nr of customers at station i.  
    """
    I = n1max    
    J = n3max
    c = 1/(mu1+mu3)
    p1 = c*mu1; p3 = c*mu3
    A = 100 * np.ones((I+1,J+I+1))
    A[0, :] = np.array(range(J+I+1))/mu3 
    
    #recursive definition in thesis
    for j in range(1, I+1):
        A[j,0] = A[j-1,1]+1/mu1 
        for i in range(1,(J+I+ 1 - j)):
            A[j,i] = c + p1 * A[j-1,i+1] + p3* A[j, i-1]
    return A


def expected_waiting_time(params, states_max):
    [n1max, n2max, n3max, n4max ] = states_max
    [l, mu1,mu2,mu3,mu4] = params
    
    A = tandem_wait_times(mu1,mu3,n1max,n3max)
    B = tandem_wait_times(mu2, mu4, n2max,n4max)

    return A,B


def wait_times_starvation(n1,n3,mu1,mu3):
    """
    input: number of peopl in tandem(n1 in first station, n3 in second station),
    service rates mu1 and mu3 
    output: the approximate expected waiting time
    """
    
    def p_L(l):
        p1 = mu1/(mu1+mu3)
        p3 = mu3/(mu1+mu3)
        res = p1**(n1+1) * p3**(n1+n3-l) * math.comb(2*n1+n3-l,n1)
        return res
    
    EW = n1/mu1 + sum(l/mu3*p_L(l) for l in range(n1+n3+1))
    return EW



def wait_times_deter(n1,n3,mu1,mu3):
    
    """
    input: number of peopl in tandem(n1 in first station, n3 in second station),
    service rates mu1 and mu3 
    output: the approximate expected waiting time
    """
    
    if n1==0 and n3 == 0:
        EW13 = 0
    elif mu1>=mu3 and n1>=1 and n3==0:
        EW13 = n1/mu3
    elif mu1>=mu3 and n3>=1:
        EW13 = (n1+n3)/mu3 - 1/mu1  
        
    elif mu1<mu3 and n1>=1 and n3==0:
        if  np.ceil(1+(n1)*(1-mu3/mu1))<=0:
            EW13 = n1/mu1
        else:
            EW13 = n1/mu3
    elif mu1<mu3 and n3>= 1:
        if np.ceil(n3+(n1+1)*(1-mu3/mu1))<=0:
            EW13 = n1/mu1
        else: 
            EW13 = (n1+n3)/mu3-1/mu1
    return EW13

#%%
"""
This part contains the code for a simulation of the simplified network. 
"""

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

def simulate_network(parameters, simulation_time,warm_up_time,  seed, strat):
    
    """
    strat = 0 for unobservable, 1 for wjsq, 2 for discounting starvation, 3 for deterministicand 4 for exact Nash
    """
    
    random.seed(seed)
    np.random.seed(seed)
    
    [lambda_rate, mu1, mu2, mu3 , mu4] = parameters
        
    time = 0
    event_queue = [] #queue which stores events
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

    departures = [0,0,0,0] #nr of departures from each queue
    
    A, B = expected_waiting_time(parameters, [200,200,200,200]) 
    #matrices which store the expected waiting times, A stores the waiting times
    # in top tandem, B the exp.waiting times in bottom tandem
    

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
    
    decision = {} #dictionary storing the decision of each customer
    
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
    
    #add the first event to the event queue 
    heapq.heappush(event_queue, Event(ARRIVAL_SYSTEM, inter_arrival_time, customer_id))
    
    #proposed decisions of each customer according to the different strategies
    dec_random ={}
    dec_nash = {}
    dec_starve = {}
    dec_deterministic = {}
    dec_wjsq = {}
    
    
    while time < simulation_time:
        event = heapq.heappop(event_queue) #get event
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
            
            if time>warm_up_time:
                state_upon_arrival[event.customer_id] = [n1,n2,n3,n4]

            Q1orQ2allstrat= [] #stores all 5 decisions

            """
            unobservable strategy
            """
            
            Q1orQ2 =  random.choices([1,2], weights=[0.5, 0.5])[0]
            Q1orQ2allstrat.append(Q1orQ2)

            if time>warm_up_time:
                dec_random[event.customer_id] = Q1orQ2
                

            """
            naive strategy
            """
            if n1/mu1 != n2/mu2:
                Q1orQ2 = np.argmin([n1/mu1,n2/mu2]) + 1 #join shortest queue
            else:
                Q1orQ2 = random.choices([1,2], weights=[0.5, 0.5])[0]
            
            Q1orQ2allstrat.append(Q1orQ2)
            if time>warm_up_time:
                dec_wjsq[event.customer_id] = Q1orQ2

            """
            strategy that discounts starvation
            """
            EW13 = wait_times_starvation(n1,n3, mu1, mu3)
            EW24 = wait_times_starvation(n2,n4,mu2,mu4)
            if EW13 == EW24:
                Q1orQ2 = random.choices([1,2], weights=[0.5, 0.5])[0]
            else:
                Q1orQ2 = np.argmin([EW13,EW24]) + 1    
            Q1orQ2allstrat.append(Q1orQ2)

            if time>warm_up_time:
                dec_starve[event.customer_id] = Q1orQ2

            """
            deterministic
            """
            if mu1<mu3 and n3<=(mu3/mu1+1)*(n1+1):
                EW13 = n1/mu1
            else: 
                EW13 = (n3+n1)/mu3-1/mu1
                
            if mu2<mu4 and n4<=(mu4/mu2+1)*(n2+1):
                EW24 = n2/mu2
            else:
                EW24 = (n4+n2)/mu4-1/mu2 
                
            if EW13 == EW24:
                Q1orQ2 = random.choices([1,2], weights=[0.5, 0.5])[0]
            else: 
                Q1orQ2 = np.argmin([EW13,EW24]) + 1
            Q1orQ2allstrat.append(Q1orQ2)

            if time>warm_up_time:
                dec_deterministic[event.customer_id] = Q1orQ2
                
            """
            # Nash strategy
            # """
            EW13 = A[n1+1,n3] - 1/mu1 - 1/mu3
            EW24 = B[n2+1,n4] - 1/mu2 - 1/mu4

            if EW13 == EW24:
                Q1orQ2 = random.choices([1,2], weights=[0.5, 0.5])[0]
            else: 
                Q1orQ2 = np.argmin([EW13,EW24]) + 1
            Q1orQ2allstrat.append(Q1orQ2)

            if time>warm_up_time:
                dec_nash[event.customer_id] = Q1orQ2
                
                
            #select the strategy of choice    
            Q1orQ2 = Q1orQ2allstrat[strat]
                
            if time>warm_up_time:
                decision[event.customer_id] = Q1orQ2 

            if Q1orQ2 == 1 and time>warm_up_time:
                arrival_times_q1[event.customer_id] = time
                
                if not server1_busy: 
                    #schedule service for the customer
                    service_time = np.random.exponential(1/mu1)
                    service_times_q1[event.customer_id] = service_time
                    heapq.heappush(event_queue, Event(DEPARTURE_Q1,time+service_time, event.customer_id))
                    server1_busy = True
                else: 
                    #add customer to queue
                    queue1.append(event.customer_id)
                    if len(queue1) > max_queue_length_1:
                        max_queue_length_1 = len(queue1) #new max length
                    
            elif Q1orQ2==2 and time>warm_up_time:
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
   
            if arrival_times_q1[event.customer_id] > warm_up_time:     
                waiting_time_q1 = time - (arrival_times_q1[event.customer_id] + service_times_q1[event.customer_id] )
                waiting_times_q1[event.customer_id] = waiting_time_q1
            
            if time>warm_up_time:
                total_waiting_time_q1 += waiting_time_q1
                departures[0] += 1
                
            if queue1:
                # schedule service for the next customer in Queue 1
                next_customer = queue1.pop(0)
                service_time = np.random.exponential(1 / mu1)
                service_times_q1[next_customer] = service_time
                heapq.heappush(event_queue, Event(DEPARTURE_Q1, time + service_time, next_customer))      
            else:
                server1_busy = False  # No one left in Queue 1
            
            arrival_times_q3[event.customer_id] = time #customer arrives to queue 3
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
            
            if arrival_times_q2[event.customer_id] > warm_up_time:     
                waiting_time_q2 = time -( arrival_times_q2[event.customer_id] + service_times_q2[event.customer_id]) 
                waiting_times_q2[event.customer_id] = waiting_time_q2
                departures[1] += 1
                total_waiting_time_q2 += waiting_time_q2
            
            
            
            if queue2:
                next_customer = queue2.pop(0)
                service_time = 1/mu2 #np.random.exponential(1 / mu2)
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

            if arrival_times_q1[event.customer_id] > warm_up_time:
                waiting_time_q3 = time - (arrival_times_q3[event.customer_id] + service_times_q3[event.customer_id])
                waiting_times_q3[event.customer_id] = waiting_time_q3
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
            
            if arrival_times_q2[event.customer_id] > warm_up_time:
                waiting_time_q4 = time - (arrival_times_q4[event.customer_id] + service_times_q4[event.customer_id]) 
                waiting_times_q4[event.customer_id] = waiting_time_q4
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
    
    
    avg_wait= np.mean([v for k,v in total_wait_time.items()])
    
    
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
        'avg_wait': avg_wait,
        'max_queue_1': max_queue_length_1,
        'max_queue_2': max_queue_length_2,
        'max_queue_3': max_queue_length_3,
        'max_queue_4': max_queue_length_4,
        "total_wait_time": total_wait_time,
        "dec_nash": dec_nash, 
        "dec_wjsq": dec_wjsq,
        "dec_deterministic": dec_deterministic,
        "dec_starve": dec_starve,
        "A": A, 
        "B": B
        }

#example usage

lam = 1.5
mu1 = 1
mu2 = 1
mu3 = 1
mu4 = 1

params=[lam,mu1,mu2,mu3, mu4]

simulation_time = 500000
warm_up_time = 10000

results = simulate_network(params, simulation_time, warm_up_time , seed=2, strat = 1)
nr_cust_13 = results["departures"][2]
nr_cust_24 = results["departures"][3]

dec_nash = results["dec_nash"]
dec_deter = results["dec_deterministic"]


# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:06:46 2024

@author: barba

Note: if time<warm up time, dont record anything, so states, wait times etc. thwy did not happen in
steady state so should not be in the analysis
"""

import os
my_path = os.path.abspath('C:\\Users\\barba\\Desktop\\thesis\\python')
os.chdir(my_path)

import numpy as np
import heapq
import random
import SimplifiedNetwork as sn
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 30
plt.rcParams['font.family'] = 'DejaVU Sans'
plt.rcParams['font.style'] = 'normal'



#%%

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

    departures = [0,0,0,0]
    
    A, B = sn.expected_waiting_time(parameters, [200,200,200,200])

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
    
    dec_random ={}
    dec_nash = {}
    dec_starve = {}
    dec_deterministic = {}
    dec_wjsq = {}
    ties_nash = {}
    ties_deter = {}
    
    
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
            
            if time>warm_up_time:
                state_upon_arrival[event.customer_id] = [n1,n2,n3,n4]

            Q1orQ2allstrat= []


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
            EW13 = sn.wait_times_starvation(n1,n3, mu1, mu3)
            EW24 = sn.wait_times_starvation(n2,n4,mu2,mu4)
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
                Q1orQ2 = 1 #random.choices([1,2], weights=[0.5, 0.5])[0]
                tie = 1
            else: 
                Q1orQ2 = np.argmin([EW13,EW24]) + 1
                tie = 0
            Q1orQ2allstrat.append(Q1orQ2)

            if time>warm_up_time:
                dec_deterministic[event.customer_id] = Q1orQ2
                ties_deter[event.customer_id] = tie
                
            """
            # Nash strategy
            # """
            EW13 = A[n1+1,n3] - 1/mu1 - 1/mu3
            EW24 = B[n2+1,n4] - 1/mu2 - 1/mu4

            if EW13 == EW24:
                Q1orQ2 = random.choices([1,2], weights=[0.5, 0.5])[0]
                tie = 1
            else: 
                Q1orQ2 = np.argmin([EW13,EW24]) + 1
                tie = 0 
            Q1orQ2allstrat.append(Q1orQ2)

            if time>warm_up_time:
                dec_nash[event.customer_id] = Q1orQ2
                ties_nash[event.customer_id] = tie
                
            Q1orQ2 = Q1orQ2allstrat[strat]
                
            if time>warm_up_time:
                decision[event.customer_id] = Q1orQ2 

        
            if Q1orQ2 == 1 and time>warm_up_time:
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
            """
            EVENT: Departure from Queue 1 + make a choice of which queue to go next
            
            if queue1 not empty:
                get new customer from the queue, add their departure to event list
                
            """
            if arrival_times_q1[event.customer_id] > warm_up_time:     
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
            
            if arrival_times_q2[event.customer_id] > warm_up_time:     
                waiting_time_q2 = time -( arrival_times_q2[event.customer_id] + service_times_q2[event.customer_id]) # add minus  service time of customer
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
        "ties_nash": ties_nash,
        "ties_deter": ties_deter,
        "A": A, 
        "B": B
        }

# Parameters
lam = 1

mu1 = 1
mu2 = 1
mu3 = 1
mu4 = 1

params=[lam,mu1,mu2,mu3, mu4]

simulation_time = 30000
warm_up_time = 1000

results = simulate_network(params, simulation_time, warm_up_time , seed=2, strat = 1)
nr_cust_13 = results["departures"][2]
nr_cust_24 = results["departures"][3]

dec_nash = results["dec_nash"]


#%%

dec_deter = results["dec_deterministic"]
states = results["state_upon_arrival"]
wt = results["waiting_times_q1"]

for i in states.keys():
    print(wt[i])

#%%
dec_nash = results["dec_nash"]
dec_deter = results["dec_deterministic"]
dec_starve = results["dec_starve"]

print(len([k for k,v in dec_nash.items() if v==dec_deter[k]])/len(dec_deter))
print(len([k for k,v in dec_nash.items() if v==dec_starve[k]])/len(dec_deter))
#%%




agreed = [k for k,v in dec_nash.items() if v==dec_deter[k]]
disagreed = [k for k,v in dec_nash.items() if v!=dec_deter[k]]
print(len(agreed)/(len(agreed)+len(disagreed)))
states_agreed = [results["state_upon_arrival"][i] for i in agreed]
states_disagreed = [results["state_upon_arrival"][i] for i in disagreed]
print(np.mean([s[2]+s[3] for s in states_agreed]))
print(np.mean([s[2]+s[3] for s in states_disagreed]))



#%%
state_upon_arrival = results["state_upon_arrival"]
waiting_times_q1 = results["waiting_times_q1"]; waiting_times_q2 = results["waiting_times_q2"];
waiting_times_q3 = results["waiting_times_q3"]; waiting_times_q4 = results["waiting_times_q4"];

total_wait_times = results["total_wait_time"]
A= results["A"]; B=results["B"]

np.mean([total_wait_times[k] for k,v in state_upon_arrival.items() if v==[2,14,20,9]])
joining_state = [2,14,20,9]

mu1 = 1.0
mu3 = 1.0
n1 = 2
n3 = 20
    
#%%
def simulate(mu1,mu2,n1,n2,nr_sim):
    """
    simulates the upper and lower stream of network many times
    """
    arrival_system=0
    departure_q1 = 1
    departure_q2=2
    
    clearance_time = np.zeros(nr_sim)
    
    for j in range(nr_sim):
        # Parameters
        np.random.seed(j)
        
        event_queue = []
        current_time = 0.0
        queue1 = []
        queue2 = []
    
        service_time = 0
        s2busy = False
        s1busy=False
        
        for i in range(n2):
            heapq.heappush(event_queue, Event(departure_q1,current_time,i))
            
        for k in range(n2, n2+n1):
            heapq.heappush(event_queue, Event(arrival_system,current_time,k))
        
        while event_queue:
            event = heapq.heappop(event_queue)
            event_time, event_type = event.event_time, event.event_type
            current_time = event_time
            
            if event.event_type== arrival_system:
                
                if not s1busy: 
                    service_time = np.random.exponential(1/mu1)
                    heapq.heappush(event_queue, Event(departure_q1,current_time+service_time, event.customer_id))
                    s1busy = True
                else: 
                    queue1.append(event.customer_id)
    
            
            if event_type == departure_q1:
    
                if queue1:
                    next_customer = queue1.pop(0)
                    service_time = np.random.exponential(1 / mu1)
                    heapq.heappush(event_queue, Event(departure_q1, current_time + service_time, next_customer))      
                else:
                    s1busy = False  
                
                if not s2busy:
                    service_time = np.random.exponential(1 / mu2)
                    heapq.heappush(event_queue, Event(departure_q2,current_time + service_time, event.customer_id))
                    s2busy = True
                else:
                    queue2.append(event.customer_id)
          
            elif event.event_type == departure_q2:
                
                if queue2:
                    next_customer = queue2.pop(0)
                    service_time = np.random.exponential(1 / mu2)
                    heapq.heappush(event_queue, Event(departure_q2, current_time + service_time, next_customer))
                else:
                    s2busy = False  
                    
        clearance_time[j] = current_time
        
    return np.mean(clearance_time)

mu1 = 1.
mu2 = 1.
mu3 = 1.
mu4 = 2.
n1 = 0
n2 = 3
n3 = 0
n4 = 4
nr_sim = 1000000

print(simulate(mu2,mu4,n2,n4,nr_sim), B[n2,n4])

#%%



dec_wjsq = results["dec_wjsq"]
dec_deterministic = results["dec_deterministic"]
dec_starve = results["dec_starve"]

state_upon_arrival = results["state_upon_arrival"]

print("mu/lam = {0}, acc of wjsq={1}".format(mu1/lam,round(len([k for k,v in dec_nash.items() if v == dec_wjsq[k]])/len(dec_nash),2)))
print("mu/lam = {0}, acc of dete={1}".format(mu1/lam,round(len([k for k,v in dec_nash.items() if v == dec_deterministic[k]])/len(dec_nash),2)))

"""
300 000 sims 
CASE: lam=1, mui=mu

mu/lam = 0.55, acc of wjsq=0.59
mu/lam = 0.55, acc of dete=0.69

mu/lam = 1.0, acc of wjsq=0.72
mu/lam = 1.0, acc of dete=0.74

mu/lam = 5.0, acc of wjsq=0.58
mu/lam = 5.0, acc of dete=0.64

message: wjsq worse than deterministic, deterministic best when mu/lam=1
i thought deterministic would get better with larger mu 

CASE: lam = 1., mu1=mu2 = 1, mu3=mu4=0.6

mu/lam = 1.0, acc of wjsq=0.60
mu/lam = 1.0, acc of dete=0.81

dewjsq gets worse because focus should be on later servers, deterministic gets better

CASE: lam = 1.
mu1 = 2*C
mu2 = 2*C 
mu3 = C
mu4 = C

C=0.55, acc of wjsq=0.58 
C=0.55, acc of dete=0.83

C=1, acc of wjsq=0.63 
C=1, acc of dete=0.78

CASE: lam = 1.
C=0.55
mu1 = C
mu2 = C 
mu3 = 2*C
mu4 = 2*C

C=0.55 acc of wjsq=0.82 
C=0.55 acc of dete=0.82

C=1.0, acc of wjsq=0.75
C 1.0, acc of dete=0.75

"""




#%%
states1 = [state_upon_arrival[k] for k,v in dec_nash.items() if v == dec_deterministic[k]]
states2= [state_upon_arrival[k] for k,v in dec_nash.items() if v != dec_deterministic[k]]

for i in range(4):
    print(np.mean([l[i] for l in states1]), np.mean([l[i] for l in states2])) #it is not true that deterministic performs better in more busy states
print("")


print(np.mean([sum(l) for l in states1]), np.mean([sum(l) for l in states2])) #it is not true that deterministic performs better in more busy states
#%%
params = [0.5,1,1,1,1]
simulation_time = 500000
warm_up_time = 10000
nr_sim =25

ci_wt = np.zeros([5,2])

for j in range(5):

    avg_waits = np.zeros(nr_sim)
    for i in range(nr_sim):
        if (i+1)%5 ==0:
            print(i)
        results = simulate_network(params,  simulation_time, warm_up_time, seed=i, strat = j)
        avg_waits[i] = results["avg_wait"]
        
    mean1=np.mean(avg_waits, axis= 0)
    sd1=np.sqrt(np.var(avg_waits, axis= 0))
    se1=2.086*sd1/np.sqrt(nr_sim)
    ci_wt[j,:] = [round(mean1-se1,3), round(mean1+se1,3)]
    print(j,[round(mean1-se1,3), round(mean1+se1,3)])
    



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






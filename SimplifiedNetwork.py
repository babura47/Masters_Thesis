# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:49:43 2024

@author: barba
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools


def expected_waiting_time(params, states_max):
    [n1max, n2max, n3max, n4max ] = states_max
    [l, mu1,mu2,mu3,mu4] = params
    
    A = tandem_wait_times(mu1,mu3,n1max,n3max)
    B = tandem_wait_times(mu2, mu4, n2max,n4max)

    return A,B

def tandem_wait_times(mu1,mu3, n1max,n3max):
    """
    returns a matrix A of size n1max by n3max such that A[i,j] = E(W|take route 1-3, n1=i, n3=j). 
    ni is the nr of customers in queue and server i.  
    """
    I = n1max    
    J = n3max
    c = 1/(mu1+mu3)
    p1 = c*mu1; p3 = c*mu3
    A = 100 * np.ones((I+1,J+I+2))
    A[0, :] = np.array(range(J+I+2))/mu3
    A[1,0] = A[0,1]+1/mu1
    
    for j in range(1, I+1):
        A[j,0] = A[j-1,1]+1/mu1 
        for i in range(1,(J+I+ 2 - j)):
            A[j,i] = c + p1 * A[j-1,i+1] + p3* A[j, i-1]
    
    return A
    
# %%
"""
plots with n3 and n4 fixed
"""
n=10 #nimax for all i

params = [1.,10.,1.,1.,1.]

n1r = np.array(range(n))
n2r = np.array(range(n))

best_resp = np.zeros([n,n])
A,B = expected_waiting_time(params, [n,n,n,n])
n3 = 0
n4 = 0
for i in range(n):
    n1 = n1r[i]
    for j in range(n):
        n2 = n2r[j]
        EW13 = A[n1,n3]
        EW24 = B[n2,n4]
        if EW13 == EW24:
            best_resp[i,j] = 1.5 # indicates both routes optimal
        else: 
            best_resp[i,j] = np.argmin([EW13,EW24]) + 1
    

# Creating a meshgrid to plot
Y,X = np.meshgrid(n1r, n2r)

# Plotting
plt.figure(figsize=(6, 6))
# Plot red dots where M is 2
plt.scatter(X[best_resp==2], Y[best_resp==2], color='red', label='Route 2-4')
# Plot blue dots where M is 1
plt.scatter(X[best_resp==1], Y[best_resp==1], color='blue', label='Route 1-3')
# Plot green dots where M is 2
plt.scatter(X[best_resp==1.5], Y[best_resp==1.5], color='yellow', label='Both routes optimal')


plt.xlabel('$n_1$',fontsize=14)
plt.ylabel('$n_2$',fontsize=14)
plt.legend(fontsize=14)
plt.title('States $(n_1,n_2,n_3={1},n_4={2})$ upon joining the system\n and the Nash strategy in each state\n[$\lambda,\mu_1, \mu_2, \mu_3, \mu_4$] = {0}'.format(params,n3,n4),fontsize=14)
plt.grid(True)
plt.show()


# %%
"""
plots with n1 and n2 fixed
"""
n=20 #nimax for all i

params = [1.,2.,1.,1.,1.]

n3r = np.array(range(n))
n4r = np.array(range(n))

best_resp = np.zeros([n,n])
A,B = expected_waiting_time(params, [n,n,n,n])
n1 = 0
n2 = 0
for i in range(n):
    n3 = n3r[i]
    for j in range(n):
        n4 = n4r[j]
        EW13 = A[n1,n3]
        EW24 = B[n2,n4]
        if EW13 == EW24:
            best_resp[i,j] = 1.5 # indicates both routes optimal
        else: 
            best_resp[i,j] = np.argmin([EW13,EW24]) + 1
    

# Creating a meshgrid to plot
Y,X = np.meshgrid(n3r, n4r)

# Plotting
plt.figure(figsize=(6, 6))
# Plot red dots where M is 2
plt.scatter(X[best_resp==2], Y[best_resp==2], color='red', label='Route 2-4')
# Plot blue dots where M is 1
plt.scatter(X[best_resp==1], Y[best_resp==1], color='blue', label='Route 1-3')
# Plot green dots where M is 2
plt.scatter(X[best_resp==1.5], Y[best_resp==1.5], color='yellow', label='Both routes optimal')


plt.xlabel('$n_3$',fontsize=14)
plt.ylabel('$n_4$',fontsize=14)
plt.legend(fontsize=14)
plt.title('States $(n_1 = {1},n_2={2},n_3,n_4)$ upon joining the system\n and the Nash strategy in each state\n[$\lambda,\mu_1, \mu_2, \mu_3, \mu_4$] = {0}'.format(params,n1,n2),fontsize=14)
plt.grid(True)
plt.show()

# %%
n=20 #nimax for all i

params = [1.,2., 2., 1., 1.]
[l,mu1,mu2,mu3,mu4] = params

n1r = np.array(range(n))
n2r = np.array(range(n))
n3r = np.array(range(n))
n4r = np.array(range(n))

states = list(itertools.product(n1r,n2r,n3r,n4r))

best_resp = {}
A,B = expected_waiting_time(params, [n,n,n,n])

for state in states:
    [n1,n2,n3,n4] = state
    
    EW13 = A[n1,n3]
    EW24 = B[n2,n4]
    if EW13 == EW24:
        best_resp[state] = 1.5 # indicates both routes optimal
    else: 
        best_resp[state] = np.argmin([EW13,EW24]) + 1

a = 0

# n1/mu1<n2/mu2 and n3/mu3<n4/mu4
new_states = [s for s in states if  s[0]/mu1 <= s[1]/mu2  and (s[2]/mu3 <= s[3]/mu4)]
strat1 = [best_resp[s] for s in new_states].count(1)
strat2 = [best_resp[s] for s in new_states].count(2)
print([strat1, strat2]) #If true, strategy works
deviates = [s for s in new_states if best_resp[s] == 2]
a+= strat1+strat2

# n1/mu1>n2/mu2 and n3/mu3>n4/mu4
new_states = [s for s in states if  s[0]/mu1 >= s[1]/mu2  and (s[2]/mu3 >= s[3]/mu4)]
strat1 = [best_resp[s] for s in new_states].count(1)
strat2 = [best_resp[s] for s in new_states].count(2)
print([strat1, strat2]) #If true, strategy works
deviates = [s for s in new_states if best_resp[s] ==1]
a+= strat1+strat2

# n1/mu1>n2/mu2 and n3/mu3< n4/mu4 and ...
new_states = [s for s in states if  s[0]/mu1 >= s[1]/mu2  and (s[2]/mu3 <= s[3]/mu4)]
new_states = [s for s in new_states if s[0]/mu1 - s[1]/mu2 + (s[2]/mu3 - s[3]/mu4) >= 0]
strat1 = [best_resp[s] for s in new_states].count(1)
strat2 = [best_resp[s] for s in new_states].count(2)
print([strat1, strat2]) #If true, strategy works
deviates = [s for s in new_states if best_resp[s] ==1]
a+= strat1+strat2


# n1/mu1<n2/mu2 and n3/mu3>n4/mu4 
new_states = [s for s in states if  s[0]/mu1 <= s[1]/mu2  and (s[2]/mu3 >= s[3]/mu4)]
new_states = [s for s in new_states if s[0]/mu1 - s[1]/mu2 + (s[2]/mu3 - s[3]/mu4) <= 0]
strat1 = [best_resp[s] for s in new_states].count(1)
strat2 = [best_resp[s] for s in new_states].count(2)
print([strat1, strat2]) #If true, strategy works
deviates = [s for s in new_states if best_resp[s] ==1]
a+= strat1+strat2





print(n**4 - a)

# when we switch if first or second stage servers are faster, results dont hold/
# then 






# %%
# # n1<n2 and n3<n4
# new_states = [s for s in states if  s[0] < s[1]  and (s[2] < s[3])]
# strat1 = [best_resp[s] for s in new_states].count(1)
# strat2 = [best_resp[s] for s in new_states].count(2)
# print([strat1, strat2]) #If true, strategy works
# deviates = [s for s in new_states if best_resp[s] == 2]


# # n1<n2 and n3>n4
# new_states = [s for s in states if  (s[0]- s[1]) < 0 and (s[2]- s[3]) > 0]

# # n1 -n2 + n3 - n4 < 0 
# strat1 = [best_resp[s] for s in new_states if  s[0] - s[1] + (s[2]- s[3]) < 0 ].count(1)
# strat2 = [best_resp[s] for s in new_states if  s[0] - s[1] + (s[2]- s[3]) < 0 ].count(2)
# print([strat1, strat2]) #If true, strategy works

# # n1 -n2 + n3 - n4 > 0 
# strat1 = [best_resp[s] for s in new_states if  s[0] - s[1] + (s[2]- s[3]) > 0].count(1)
# strat2 = [best_resp[s] for s in new_states if  s[0] - s[1] + (s[2]- s[3]) > 0].count(2)
# print([strat1, strat2]) #If true, strategy works







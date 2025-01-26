# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:49:43 2024

@author: barba
"""
import numpy as np
import matplotlib.pyplot as plt
import math


#verify it's optimal with simulation

def expected_waiting_time(params, states_max):
    [n1max, n2max, n3max, n4max ] = states_max
    [l, mu1,mu2,mu3,mu4] = params
    
    A = tandem_wait_times(mu1,mu3,n1max,n3max)
    B = tandem_wait_times(mu2, mu4, n2max,n4max)

    return A,B

def tandem_wait_times(mu1,mu3, n1max,n3max):
    """
    returns a matrix A of size n1max by n3max+n1max such that A[i,j] = E(clearing time, n1=i, n2=j). 
    ni is the nr of customers in queue and server i.  
    """
    I = n1max    
    J = n3max
    c = 1/(mu1+mu3)
    p1 = c*mu1; p3 = c*mu3
    A = 100 * np.ones((I+1,J+I+1))
    A[0, :] = np.array(range(J+I+1))/mu3
    
    for j in range(1, I+1):
        A[j,0] = A[j-1,1]+1/mu1 
        for i in range(1,(J+I+ 1 - j)):
            A[j,i] = c + p1 * A[j-1,i+1] + p3* A[j, i-1]
    return A




def wait_times_starvation(n1,n3,mu1,mu3):

    def p_L(l):
        p1 = mu1/(mu1+mu3)
        p3 = mu3/(mu1+mu3)
        res = p1**(n1+1) * p3**(n1+n3-l) * math.comb(2*n1+n3-l,n1)
        return res
    
    EW = n1/mu1 + sum(l/mu3*p_L(l) for l in range(n1+n3+1))
    return EW



def wait_times_deter(n1,n3,mu1,mu3):
    
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

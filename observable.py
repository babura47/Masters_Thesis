# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:30:56 2024

@author: barba
"""
import numpy as np
from itertools import product

def clearance_times(params, n_i_max):
    """
    returns a matrix A of size n1max by n3max such that A[i,j] = E(clearing time, n1=i, n2=j). 
    ni is the nr of customers in queue and server i.  
    """
    
    [mu1,mu2,mu3,mu4,mu5] = params
    [n1max,n2max,n3max,n4max,n5max] = n_i_max
    
    A = 1000*np.ones((n1max+1,n2max+1,n3max+n1max+1,n4max+n1max+n2max+1,n5max+n2max+1))
    
    irange = range(n1max+1)
    jrange = range(n2max+1)
    krange = range(n3max+n1max+1)
    lrange = range(n4max+n1max+n2max+1)
    mrange = range(n5max+n2max+1)
    
    A[0,0,:,0,0] = np.array(krange)/mu3
    A[0,0,0,:,0] = np.array(lrange)/mu4
    A[0,0,0,0,:] = np.array(mrange)/mu5
    
    it=product(krange, lrange, mrange)
        
    for i in [i for i in it if len(np.nonzero(i)[0])>1]:
        c= 1/(int(i[0]>0)*mu3+int(i[1]>0)*mu4+int(i[2]>0)*mu5) 
        A[0,0,i[0],i[1],i[2]] = c + int(i[0]>0)*mu3*c*A[0,0,i[0]-1,i[1],i[2]] + \
            int(i[1]>0)*mu4*c*A[0,0,i[0],i[1]-1,i[2]] + int(i[2]>0)*mu5*c*A[0,0,i[0],i[1],i[2]-1]
    
    it = product(irange,jrange,krange,lrange,mrange)
    
    for i in [i for i in it if (i[0]>0 or i[1]>0) and (i[0]+i[2]<=n1max+n3max and i[0]+i[1]+i[3]<=n4max+n1max+n2max and i[1]+i[4]<=n5max+n2max)]:
        c= 1/(int(i[0]>0)*mu1+int(i[1]>0)*mu2+int(i[2]>0)*mu3+int(i[3]>0)*mu4+int(i[4]>0)*mu5)  
        
        A[i[0],i[1],i[2],i[3],i[4]] = c + c*(int(i[2]>0)*mu3)*A[i[0],i[1],i[2]-1,i[3],i[4]]+\
        c*(int(i[3]>0)*mu4)*A[i[0],i[1],i[2],i[3]-1,i[4]] + c*(int(i[4]>0)*mu5)*A[i[0],i[1],i[2],i[3],i[4]-1]
                
        if i[0]>0 and i[2]/mu3<i[3]/mu4:
            A[i[0], i[1],i[2],i[3],i[4]] += c*mu1*A[i[0]-1,i[1],i[2]+1,i[3],i[4]]  
        elif i[0]>0 and i[2]/mu3>= i[3]/mu4:
            A[i[0], i[1],i[2],i[3],i[4]] += c*mu1*A[i[0]-1,i[1],i[2],i[3]+1,i[4]] 
            
        if i[1]>0 and i[3]/mu4< i[4]/mu5:
            A[i[0], i[1],i[2],i[3],i[4]] += c*mu2*A[i[0],i[1]-1,i[2],i[3]+1,i[4]] 
        elif i[1]>0 and i[3]/mu4>= i[4]/mu5:
            A[i[0], i[1],i[2],i[3],i[4]] += c*mu2*A[i[0],i[1]-1,i[2],i[3],i[4]+1] 
            
    return A
    

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:47:27 2024

@author: barba

This file contains the code which is used to compute the unobservable Nash equilibrium
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import least_squares

max_res_q1=1e-5
max_res_r0=1e-5
max_res_nonzero = 1e-5

tol_E_q1 = 1e-3
tol_E_r0=1e-3
tol_E_nonzero=1e-3




def nash_nonzero_anal(params):
    """
    takes the parameters and gives a analytical solution of nash 
    """
    [l,a,b,c,d,e] = params  #lam, mu1,mu2,mu3,mu4,mu5
    # Basic feasibility
    check0 = l < a + b
    check01 = l < c + d + e

    feasibility = [check0, check01]
    if all(feasibility)  != True:
        print('the system is not feasible!')
    
    B = -(a - b) * l - a**2 - b**2
    A = (a - b) * l
    C = a**2

    D = B**2 - 4 * A * C

    if a != b:
        p = (-B - np.sqrt(D)) / (2 * A)
    else:
        p = 0.5
        

    # terms that make up the 4th degree polynomial in q that characterises Nash equilibrium
    T1 = l**2 * p**4 * (c - d)**2
    T2 = 2 * p**3 * l * (2 * c * p * d + c**2 * d + d**3 - c**2 * p * l - c**3 - c * d**2 - l * d**2 * p)
    T3 = c**2 * p**4 * l**2 + p**2 * c**4 + p**2 * d**4 + l**2 * d**2 * p**4 + 4 * c**3 * p**3 * l + 2 * c * p**3 * d**2 * l +  \
    2 * (c * d * p)**2 - 2 * d**3 * p**3 * l - 2 * c * p**4 * l**2 * d - 4 * c**2 * p**3 * l * d
    T4 = 2 * p**2 * c**2 * (p * d * l - p * c * l - c**2 - d**2)
    T5 = c**4 * p**2
    T6 = p**2 * l * (c - d)
    T7 = p * (l * p * (d - c) - (d**2 + c**2))
    T8 = c**2 * p
    T9 = p * l * (p - 1) * (d - c)
    T10 = (p - 1) * c**2
    T11 = e * (p - 1) * (l - 2 * l * p - e) - d * (p - 1) * (l - 2 * l * p + d)
    T12 = p * l * (p - 1) * (e - d)
    T13 = p**2 * l**2 * d**2 + c**2 * p**2 * l**2 + (c * p**2 * l)**2 + (p**2 * l * d)**2 + 4 * c * d * p**3 * l**2 
    - 2 * c**2 * p**3 * l**2 - 2 * p**2 * l**2 * c * d - 2 * p**4 * l**2 * c * d - 2 * p**3 * l**2 * d**2
    T14 = 2 * p**3 * l * c**2 * d + 2 * p * l * c**2 * d + 4 * c**3 * p**2 * l - 2 * p**3 * c**3 * l - \
    4 * p**2 * c**2 * l * d - 2 * c**3 * p * l
    T15 = c**4 * (p - 1)**2
    T16 = p * (l * (p - 1) * (d - e) - e**2)
    T17 = p * e * (e + l * (p - 1)) + (p - 1) * d * (d - p * l)
    C = (e - d) * l * (p - 1)**2

    C0 = T5 * C + T8 * T10 * T11 + T15 * T17
    C1 = T4 * C + T7 * T10 * T11 + T8 * T9 * T11 + T8 * T10 * T12 + T14 * T17 + T15 * T16
    C2 = T3 * C + T6 * T10 * T11 + T7 * T9 * T11 + T7 * T10 * T12 + T8 * T9 * T12 + T13 * T17 + T14 * T16
    C3 = T2 * C + T6 * T9 * T11 + T7 * T9 * T12 + T13 * T16
    C4 = T1 * C + T6 * T9 * T12
    
    coefvct = [C4, C3, C2, C1, C0]

    q = np.roots(coefvct) #finds roots of poly given by coefvct
    q = np.real(q[np.isreal(q)]) #takes the real part
    
    q = q[(q >= 0) & (q <= 1)][0] #takes the part between 0 and 1
        
    #from computations
    Nr = c**2 * p - c * q * p**2 * l - p * q * c**2 + c * q**2 * p**2 * l - q * p * d**2 + l * q * d * p**2 - q**2 * p**2 * l * d
    Dr = p**2 * l * q * d - p * l * q * d - c**2 + c * q * p * l + p * c**2 - c * q * p**2 * l
    r = Nr / Dr 
    
    v = np.array([p, q,r])

    res = np.abs(F(v[1:],params)) #residual
    expectations = [E(p*l, a)[0],E((1-p)*l, b)[0],E( p * q * l,c)[0] ,
    E((p * (1 - q) + r * (1 - p)) * l,d)[0] ,E((1 - p) * (1 - r) * l,e) [0]]

    return v, res, expectations, coefvct


# expectation
def E(l,mu):
    if l<mu:
        e=l/(mu*(mu-l)) 
        feas=True
    else: 
        e = 10000
        feas=False
    return e, feas

# derivative of expectation
def dpE(l, mu):
    if l<mu:
        e=(2*l*mu-l**2)/(mu*(mu-l)**2)
    else: 
        e=10000
    return e


# social cost
def SC(params, probs): 
    [p,q,r]=probs
    [l,a,b,c,d,e]=params #lam, mu1,mu2,mu3,mu4,mu5
    E1 = E(p*l, a)[0]
    E2 = E((1-p)*l, b)[0]
    E3 = E(p*q*l, c)[0]
    E4 = E(((1-p)*r+p*(1-q))*l, d)[0]
    E5 = E((1-p)*(1-r)*l, e) [0]
    Es= [E1, E2, E3, E4, E5]
    
    
    not_feas = [i for i in range(len(Es)) if Es[i]==10000] #only the expectations which are infinite
    
    #if at least one queue blows up, set social cost to 100000
    if len(not_feas)==0:
        sc = p*l*E1 + (1-p)*l*E2 + p*q*l*E3 + ((1-p)*r+p*(1-q))*l*E4 + (1-p)*(1-r)*l*E5
    else:
        sc=10000
    
    #jacobian of social cost
    jsc= [0,0,0]
    jsc[0] = l * dpE(p*l, a)-l*dpE((1-p)*l, b) + l*q*dpE(p*q*l, c) +(l*(1-q)-l*r)*dpE(l*(p*(1-q)+(1-p)*r),d) - (1-r)*l*dpE((1-p)*(1-r)*l, e)
    jsc[1] = l*p*dpE(p*q*l, c) - l*p *dpE(l*(p*(1-q)+(1-p)*r),d) 
    jsc[2] = dpE(l*(p*(1-q)+(1-p)*r),d) *l*(1-p) - l*(1-p)* dpE((1-p)*(1-r)*l, e)
    return sc, jsc, not_feas, [E1,E2,E3,E4,E5]



def social_optimum_scipy(params, x0):
    
    def SC_in_x(probs):
        return SC(params, probs)[0]
    
    def SC_jacob(probs):    
        return SC(params, probs)[1]
    
    [p,q,r] =  x0
    [l,a,b,c,d,e] = params #lam, mu1,mu2,mu3,mu4,mu5
        
    bounds = Bounds([0, 0,0], [1, 1,1])
    
    res= minimize(SC_in_x, x0, bounds=bounds, tol = 1e-13, jac= SC_jacob)
    [p,q,r] = res.x
    [l,a,b,c,d,e] = params #lam, mu1,mu2,mu3,mu4,mu5
    check = res.fun>0 
    
    if not np.all([p*l<a, (1-p)*l<b, p*q*l<c, l*(p*(1-q)+r*(1-p))<d, (1-r)*(1-p)*l<e]):
        print("social optimum does not meet stability conditions")
        print(SC(params, res.x)[3])
        
    return res.x, check



def F(w, params):
    """
    takes in w=[p,q,r] and parameters (lam and mu_i) and returns
    F(x), where F(x) is the difference between the RHS and LHS of the two equations
    that characterise Nash equilbrium in the nonzero routing case
    """
    [l,a,b,c,d,e]=params #lam, mu1,mu2,mu3,mu4,mu5
    
    q = w[0]
    r = w[1]
    
    B = -(a - b) * l - a**2 - b**2
    A = (a - b) * l
    C = a**2

    D = B**2 - 4 * A * C

    if a != b:
        p = (-B - np.sqrt(D)) / (2 * A)
    else:
        p = 0.5
    
    E3 =  E(p*q*l, c)[0]
    E4 = E((p*(1-q)+r*(1-p))*l, d)[0]
    E5= E((1-r)*(1-p)*l,e)[0]
    
    eq2 = E4-E5
    eq3  = E3-E4
    
    return np.array([eq2, eq3])
    

def Fq1(w, params):
    """
    takes in w=[p,q,r] and parameters (lam and mu_i) and returns
    F(x), where F(x) is the difference between the RHS and LHS of the two equations
    that characterise Nash equilbrium in the zero routing q=1 case
    """
    [l,a,b,c,d,e]=params #lam, mu1,mu2,mu3,mu4,mu5
    p = w[0]
    r = w[1]
    
    E1=E(p*l, a)[0]
    E2 = E((1-p)*l, b)[0]
    E3 = E(p*l, c)[0]
    E4 = E(r*(1-p)*l, d)[0]
    E5= E((1-r)*(1-p)*l,e)[0]
    
    eq1 = E4-E5
    eq2= E1+E3 - E2 -E4

    return np.array([eq1, eq2]) 



def Fr0(w, params):
    """
    takes in w=[p,q,r] and parameters (lam and mu_i) and returns
    F(x), where F(x) is the difference between the RHS and LHS of the two equations
    that characterise Nash equilbrium in the zero routing r=0 case
    """
    [l,a,b,c,d,e]=params #lam, mu1,mu2,mu3,mu4,mu5
    p = w[0]
    q = w[1]
    
    E1=E(p*l, a)[0]
    E2 = E((1-p)*l, b)[0]
    E3 = E(p*q*l, c)[0]
    E4 = E((1-q)*p*l, d)[0]
    E5= E((1-p)*l,e)[0]
    
    eq1 = E3-E4
    eq2= E1+E3-E2-E5
    
    return np.array([eq1, eq2]) 


def zero_routing_q1(params, x0, tol_res, tol_e):
    """ 
    finds a minimum of function Fq1 and does some checks to see if solution is valid
    """
    [l,a,b,c,d,e]=params #lam, mu1,mu2,mu3,mu4,mu5

    #make Fq1 into a function of [p,r] only, so we can minimise with respect to that
    def Fq1_in_x(probs):
        return Fq1(probs, params)

    
    bounds = Bounds([0, 0], [1, 1])
    res= least_squares(Fq1_in_x, x0,ftol=1e-15, bounds=bounds) #res.x = [p,r]
    
    
    probs= np.insert(res.x, values=1, obj=1) #insert q=1 in the middle to get [p,q,r]
    residual = Fq1(res.x, params)


    p=probs[0]; q=probs[1]; r=probs[2]    

    check_res = np.max(abs(residual))<tol_res #residual small enough
    check_e = E(p*l, a)[0] - E((1-p)*l,b)[0] > - tol_e  #in zero routing with q=1, EW1> EW2
    check_feas = np.all([p*l<a, (1-p)*l<b, p*q*l<c, l*(p*(1-q)+r*(1-p))<d, (1-r)*(1-p)*l<e]) #solution is stable
    check = check_res and check_e and check_feas
        

    
    return probs, residual, check, E(p*l, a)[0]- E((1-p)*l,b)[0]

def zero_routing_r0(params, x0, tol_res, tol_e):
    """ 
    finds a minimum of function Fr0 and does some checks to see if solution is valid
    """
    
    def Fr0_in_x(probs):
        return Fr0(probs, params)
    
    bounds = Bounds([0, 0], [1, 1])
    res= least_squares(Fr0_in_x, x0,ftol=1e-15, bounds=bounds)
    
    probs= np.insert(res.x, values=0, obj=2)
    residual = Fr0(res.x, params)


    [l,a,b,c,d,e]=params #lam, mu1,mu2,mu3,mu4,mu5
    
    p=probs[0]; q=probs[1]; r=probs[2]    
    
    check_res = np.max(abs(residual))<tol_res
    check_e = E(p*l, a)[0] - E((1-p)*l,b)[0] < tol_e
    check_feas = np.all([p*l<a, (1-p)*l<b, p*q*l<c, l*(p*(1-q)+r*(1-p))<d, (1-r)*(1-p)*l<e])
    check = check_res and check_e and check_feas
        
    
    return probs, residual, check, E(p*l, a)[0] - E((1-p)*l,b)[0]


def nash_nonzero(params, x0, tol_res, tol_e):
    """ 
    finds a minimum of function F and does some checks to see if solution is valid
    """
    [l,a,b,c,d,e]=params    
    
    B = -(a - b) * l - a**2 - b**2
    A = (a - b) * l
    C = a**2

    D = B**2 - 4 * A * C

    if a != b:
        p = (-B - np.sqrt(D)) / (2 * A)
    else:
        p = 0.5
        
    def F_in_qr(probs):
        return F(probs, params)
    bounds = Bounds([0, 0], [ 1, 1])
    res= least_squares(F_in_qr, x0, bounds=bounds, ftol=1e-15, jac='2-point')

    
    probs= np.insert(res.x, values=p, obj=0)
    residual = F(probs[1:], params)
    [p,q,r] = probs


    check_res = np.max(abs(residual))<tol_res
    check_e =abs( E(p*l, a)[0] - E((1-p)*l,b)[0]) < tol_e
    check_feas = np.all([p*l<a, (1-p)*l<b, p*q*l<c, l*(p*(1-q)+r*(1-p))<d, (1-r)*(1-p)*l<e])
    check = check_res and check_e and check_feas
    
    
    
    return probs, residual, check, E(p*l, a)[0] -E((1-p)*l,b)[0]

def nash_solver(params,x0, tol_res, tol_e):
    
    [p,q,r] = x0
    [l,a,b,c,d,e]=params #lam, mu1,mu2,mu3,mu4,mu5
    

    sol_found = False
          
    while sol_found == False:
        
        [probs1, resid1,check1,_] = zero_routing_q1(params,x0[0:3:2],tol_res,tol_e)
        [probs2, resid2,check2,_] = zero_routing_r0(params,x0[0:2],tol_res,tol_e)
        [probs3, resid3,check3,_] = nash_nonzero(params,x0[1:],tol_res,tol_e)
        
        checks=[check1,check2,check3]
        resids= [resid1,resid2,resid3]
        
        # all three tolerances too large to count as solution
        if len([i for i in range(3) if np.all(resids[i]>1e-5)])==3:
            print("no sol found")
            
        
        if sum(checks)>=2: #too many solutions, make tolerance smaller
            tol_res=0.95*tol_res
            tol_e = 0.95*tol_e
        elif sum(checks) ==0: #no solutions, increase tolerance
            tol_res=1.05*tol_res
            tol_e = 1.05*tol_e            
        else:  #only one solution
            if check3:
                probs = probs3
                resid = resid3
                solver_ind = 3
            elif check1:
                probs = probs1
                resid = resid1
                solver_ind = 1
            elif check2:
                probs = probs2
                resid = resid2
                solver_ind = 2
            sol_found = True
    [p,q,r] = probs
    
    if not np.all([p*l<a, (1-p)*l<b, p*q*l<c, l*(p*(1-q)+r*(1-p))<d, (1-r)*(1-p)*l<e]):
        print('Nash output not feasible')

    
    return probs,resid, checks, solver_ind, resids, [tol_res, tol_e]


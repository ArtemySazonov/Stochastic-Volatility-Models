import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
import math

from dataclasses import dataclass
from typing import Union, Callable, Optional
from copy import error
from dataclasses import dataclass
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import newton
from scipy.stats import norm

import warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")

@dataclass
class HestonParameters:
    kappa:  Union[float, np.ndarray]
    gamma:  Union[float, np.ndarray]
    rho:    Union[float, np.ndarray]
    vbar:   Union[float, np.ndarray]
    v0:     Union[float, np.ndarray]
        
@dataclass
class MarketState:
    stock_price:   Union[float, np.ndarray]
    interest_rate: Union[float, np.ndarray]

def get_len_conf_interval(data:             np.ndarray, 
                          confidence_level: float = 0.05):
    """Get the confidence interval length for a given confidence level.
    Args:
        data:             The data to compute the confidence interval for.
        confidence_level: The confidence level to use.
    
    Returns:
        The confidence interval.
    """
    return -2*sps.norm.ppf(confidence_level*0.5) * np.sqrt(np.var(data) / len(data))

def mc_price(payoff:                 Callable,
             simulate:               Callable,
             market_state:           MarketState,
             params:                 HestonParameters,
             T:                      float    = 1.,
             dt:                     float    = 1e-2,
             absolute_error:         float    = 0.01,
             confidence_level:       float    = 0.05,
             batch_size:             int      = 10_000,
             MAX_ITER:               int      = 100_000,
             control_variate_payoff: Callable = None,
             control_variate_iter:   int      = 1_000,
             debug:                  bool     = False,
             **kwargs):
    """A function that performs a Monte-Carlo based pricing of a 
       derivative with a given payoff (possibly path-dependent)
       under the Heston model.

    Args:
        payoff (Callable):                  payoff function
        simulate (Callable):                simulation engine
        market_state (MarketState):         market state
        params (HestonParameters):          Heston parameters
        T (float, optional):                Contract expiration time. Defaults to 1.. 
        absolute_error (float, optional):   absolute error of the price. Defaults to 0.01 (corresponds to 1 cent). 
        confidence_level (float, optional): confidence level for the price. Defaults to 0.05.
        batch_size (int, optional):         path-batch size. Defaults to 10_000.
        MAX_ITER (int, optional):           maximum number of iterations. Defaults to 100_000.  

    Returns:    
        The price of the derivative.
              
"""

    arg = {'state':           market_state, #renamed to market_state from state
           'heston_params':   params, 
           'time':            T , 
           'dt':              dt, 
           'n_simulations':   batch_size}

    args       = {**arg, **kwargs}
    iter_count = 0   

    length_conf_interval   = 1.
    n                      = 0
    C                      = -2*sps.norm.ppf(confidence_level*0.5)
    derivative_price_array = np.array([], dtype=np.float64)
    sigma_n  = 0.
    batch_new = np.zeros(batch_size, dtype=np.float64)
    current_Pt_sum = 0.        

    # np.sqrt(np.var(data) / len(data))
    if control_variate_payoff is None:
        while length_conf_interval > absolute_error and iter_count < MAX_ITER:
            batch_new = payoff(simulate(**args)['price'])
            # derivative_price_array = np.append(derivative_price_array, batch_new)
            iter_count+=1

            sigma_n = (sigma_n*(n-1.) + np.var(batch_new)*(batch_size - 1.))/(n + batch_size - 1.)
            current_Pt_sum = current_Pt_sum + np.sum(batch_new) 

            n+=batch_size
            length_conf_interval = C * np.sqrt(sigma_n / n)

            if debug:
                print(f"Current price: {current_Pt_sum/n:.4f} +/- {length_conf_interval:.4f}")
    else:
        S = simulate(control_variate_iter)
        c = np.cov(payoff(S), control_variate_payoff(S))
        theta = c[0, 1] / c[1, 1]
        while length_conf_interval > absolute_error and iter_count < MAX_ITER:
            batch_new = payoff(simulate(**args)['price']) - theta * control_variate_payoff(simulate(**args)['price'])
            # derivative_price_array = np.append(derivative_price_array, batch_new)
            iter_count+=1

            sigma_n = (sigma_n*(n-1.) + np.var(batch_new)*(batch_size - 1.))/(n + batch_size - 1.)
            current_Pt_sum = current_Pt_sum + np.sum(batch_new) 

            n+=batch_size
            length_conf_interval = C * np.sqrt(sigma_n / n)

            if debug:
                print(f"Current price: {current_Pt_sum/n:.4f} +/- {length_conf_interval:.4f}")

    if debug:
        print(f"Number of iterations:   {iter_count}\nNumber of simulations:  {n}\nAbsolute error:         {length_conf_interval}\nConfidence level:       {confidence_level}\n")

    return current_Pt_sum/n

def simulate_heston_euler(state:           MarketState,
                          heston_params:   HestonParameters,
                          time:            float = 1.,
                          dt:              float = 1e-2,
                          n_simulations:   int = 10_000
                          ) -> dict:
    """Simulation engine for the Heston model using the Euler scheme.

    Args:
        state (MarketState): _description_
        heston_params (HestonParameters): _description_
        time (float, optional): _description_. Defaults to 1..
        dt (float, optional): _description_. Defaults to 1e-2.
        time_batch_size (int, optional): _description_. Defaults to 10_000.
        n_simulations (int, optional): _description_. Defaults to 10_000.

    Raises:
        Error: _description_

    Returns:
        dict: _description_
    """    
    if time<=0:
        raise error("Time must be bigger than 0")
    
    # initialize market and model parameters
    r, s0 = state.interest_rate, state.stock_price
    
    v0, rho, kappa, vbar, gamma = heston_params.v0, heston_params.rho, heston_params.kappa, \
                                  heston_params.vbar, heston_params.gamma
    
    vt         = np.zeros(n_simulations)
    vt[:]      = v0
    log_st     = np.zeros(n_simulations)
    log_st[:]  = np.log(s0)
    N_T        = int(time / dt)
    
    Z1         = np.random.normal(size=(n_simulations, N_T))
    Z2         = np.random.normal(size=(n_simulations, N_T))
    V          = np.zeros([n_simulations, N_T])
    V[:, 0]    = vt
    
    logS       = np.zeros([n_simulations, N_T])
    logS[:, 0] = log_st

    for i in range(0,  N_T-1):
        vmax         = np.maximum(V[:, i],0)
        S1           = (r - 0.5 * vmax) * (dt)
        S2           = np.sqrt(vmax*(dt)) * Z1[:, i]
        logS[:, i+1] = logS[:, i] + S1 + S2
        V1           = kappa*(vbar - vmax)*(dt)
        V2           = gamma*np.sqrt(vmax*(dt))*(rho*Z1[:, i]+np.sqrt(1-rho**2)*Z2[:, i])
        V[:, i+1]    = V[:, i] + V1 + V2

    vt     = V[:, N_T-1]
    log_st = logS[:, N_T-1]
        
    return {"price": np.exp(log_st), "volatility": vt}

def simulate_heston_andersen_qe(state:        MarketState,
                               heston_params: HestonParameters,
                               time:          float = 1.,
                               dt:            float = 1e-2,
                               n_simulations: int = 10_000,
                               Psi_c:         float=1.5,
                               gamma_1:       float=0.0     
                               ) -> dict: 
    """Simulation engine for the Heston model using the Quadratic-Exponential Andersen scheme.

    Args:
        state (MarketState): _description_
        heston_params (HestonParameters): _description_
        time (float, optional): _description_. Defaults to 1..
        dt (float, optional): _description_. Defaults to 1e-2.
        n_simulations (int, optional): _description_. Defaults to 10_000.
        Psi_c (float, optional): _description_. Defaults to 1.5.
        gamma_1 (float, optional): _description_. Defaults to 0.5.

    Raises:
        Error: _description_
        Error: _description_

    Returns:
        dict: _description_
    """    
    
    if Psi_c>2 or Psi_c<1:
        raise error('1<=Psi_c<=2 ')
    if gamma_1 >1 or gamma_1<0:
        raise error('0<=gamma_1<=1')
        
    gamma_2 = 1.0 - gamma_1
    
    r, s0 = state.interest_rate, state.stock_price
    v0, rho, kappa, vbar, gamma = heston_params.v0, heston_params.rho, heston_params.kappa, heston_params.vbar, heston_params.gamma
    
    
    E          = np.exp(-kappa*dt)
    K_0        = -(rho*kappa*vbar/gamma)*dt
    K_1        = gamma_1*dt*(rho*kappa/gamma - 0.5) - rho/gamma
    K_2        = gamma_2*dt*(rho*kappa/gamma - 0.5) + rho/gamma
    K_3        = gamma_1*dt*(1.0 - rho**2)
    K_4        = gamma_2*dt*(1.0 - rho**2)
    N_T        = int(time / dt)
    
    vt         = np.zeros(n_simulations)
    log_st     = np.zeros(n_simulations)
        
    V          = np.zeros([n_simulations, N_T])
    V[:, 0]    = v0

    logS       = np.zeros([n_simulations, N_T])
    logS[:, 0] = np.log(s0)

    Z          = np.random.normal(size=(n_simulations, N_T))
    Z_V = np.random.normal(size=(n_simulations, N_T))
    U = np.random.uniform(size=(n_simulations, N_T))
    # ksi = np.random.binomial(1, 1.0-p, size=(n_simulations, N_T))
    # eta = np.random.exponential(scale = 1., size=(n_simulations, N_T))
    p1 = (1. - E)*(gamma**2)*E/kappa
    p2 = (vbar*gamma**2)/(2.0*kappa)*((1-E)**2)


    for i in range(N_T - 1):
        m            = vbar+(V[:, i] - vbar)*E
        s_2          = V[:, i]*p1 + p2
        Psi          = s_2/np.power(m,2) 

        cond         = np.where(Psi<=Psi_c)
        c            = 2 / Psi[cond]
        b            = c - 1. + np.sqrt(c*(c - 1.))
        a            = m[cond]/(1.+b)
        b            = np.sqrt(b)
        # Z_V          = np.random.normal(size=cond[0].shape[0])
        V[cond, i+1] = a*(np.power(b+Z_V[cond, i] , 2))

        cond         = np.where(Psi>Psi_c)
        p            = (Psi[cond] - 1)/(Psi[cond] + 1)
        beta         = (1.0 - p)/m[cond]

        V[cond,i+1] = np.where(U[cond, i] < p, 0., np.log((1-p)/(1-U[cond, i]))/beta)


        logS[:,i+1] = logS[:,i] + r*dt+K_0 + K_1*V[:,i] + K_2*V[:,i+1] \
                        + np.sqrt(K_3*V[:,i]+K_4*V[:,i+1]) * Z[:,i]

    vt     = V[:, N_T-1]
    log_st = logS[:, N_T-1]
            
    return {"price": np.exp(log_st), "volatility": vt}


def calculate_r_for_andersen_tg(x_:    float,
                                maxiter = 2500 , 
                                tol= 1e-5
                               ) -> float:
    
    def foo(x: float):
        
        return x*norm.pdf(x) + norm.cdf(x)*(1+x**2) - (1+x_)*(norm.pdf(x) + x*norm.cdf(x))**2

    def foo_dif(x:  float):

        return norm.pdf(x) - x**2 * norm.pdf(x) + norm.pdf(x)*(1+x**2) + 2*norm.cdf(x)*x - \
                2*(1+x_)*(norm.pdf(x) + x*norm.cdf(x))*(-norm.pdf(x)*x + norm.cdf(x) + x*norm.pdf(x) )

    def foo_dif2(x:  float):
        return -x*norm.pdf(x) - 2*x* norm.pdf(x) + x**3 * norm.pdf(x) -x*norm.pdf(x)*(1+x**2) + \
                2*norm.cdf(x)*x + 2*norm.pdf(x)*x + 2*norm.cdf(x) + \
                2*(1+x_)*(-norm.pdf(x)*x + norm.cdf(x) + x*norm.pdf(x))**2 + \
                2*(1+x_)*(norm.pdf(x) + x*norm.cdf(x))*(x**2*norm.pdf(x) + norm.pdf(x) + norm.pdf(x) -x*norm.pdf(x) )


    return newton(foo,  x0 = 1/x_,fprime = foo_dif, fprime2 = foo_dif2, maxiter = maxiter , tol= tol )



def simulate_heston_andersen_tg(state:        MarketState,
                               heston_params: HestonParameters,
                               x_grid:        np.array,
                               f_nu_grid:     np.array,
                               f_sigma_grid:  np.array,
                               time:          float = 1.,
                               dt:            float = 1e-2,
                               n_simulations: int = 10_000,
                               Psi_c:         float=1.5,
                               gamma_1:       float=0.0
                               
                               ) -> dict: 
    """Simulation engine for the Heston model using the Quadratic-Exponential Andersen scheme.

    Args:
        state (MarketState): _description_
        heston_params (HestonParameters): _description_
        time (float, optional): _description_. Defaults to 1..
        dt (float, optional): _description_. Defaults to 1e-2.
        n_simulations (int, optional): _description_. Defaults to 10_000.
        Psi_c (float, optional): _description_. Defaults to 1.5.
        gamma_1 (float, optional): _description_. Defaults to 0.5.

    Raises:
        Error: _description_
        Error: _description_

    Returns:
        dict: _description_
    """     
    gamma_2 = 1.0 - gamma_1
    
    r, s0 = state.interest_rate, state.stock_price
    v0, rho, kappa, vbar, gamma = heston_params.v0, heston_params.rho, heston_params.kappa, heston_params.vbar, heston_params.gamma
    
    
    E          = np.exp(-kappa*dt)
    K_0        = -(rho*kappa*vbar/gamma)*dt
    K_1        = gamma_1*dt*(rho*kappa/gamma - 0.5) - rho/gamma
    K_2        = gamma_2*dt*(rho*kappa/gamma - 0.5) + rho/gamma
    K_3        = gamma_1*dt*(1.0 - rho**2)
    K_4        = gamma_2*dt*(1.0 - rho**2)
    N_T        = int(time / dt)
    
    vt         = np.zeros(n_simulations)
    log_st     = np.zeros(n_simulations)
        
    V          = np.zeros([n_simulations, N_T])
    V[:, 0]    = v0

    logS       = np.zeros([n_simulations, N_T])
    logS[:, 0] = np.log(s0)

    Z          = np.random.normal(size=(n_simulations, N_T))
    Z_V        = np.random.normal(size=(n_simulations, N_T))
    
    
    dx = np.diff(x_grid[0:2])[0]
    p1 = (1. - E)*(gamma**2)*E/kappa
    p2 = (vbar*gamma**2)/(2.0*kappa)*((1-E)**2)
    
    for i in range(N_T - 1):
        m            = vbar+(V[:, i] - vbar)*E
        s_2          = V[:, i]*p1 + p2
        Psi          = s_2/np.power(m,2) 
        
        
        #inx = np.searchsorted(x_grid, Psi)
        
        inx = (Psi/dx).astype(int)
        
        nu           = m*f_nu_grid[inx]
        sigma        = np.sqrt(s_2)*f_sigma_grid[inx]
        

        V[:,i+1]     = np.maximum( nu + sigma*Z_V[:,i+1], 0)


        logS[:,i+1] = logS[:,i] + r*dt+K_0 + K_1*V[:,i] + K_2*V[:,i+1] \
                        + np.sqrt(K_3*V[:,i]+K_4*V[:,i+1]) * Z[:,i]

    vt     = V[:, N_T-1]
    log_st = logS[:, N_T-1]
    
    return {"price": np.exp(log_st), "volatility": vt}

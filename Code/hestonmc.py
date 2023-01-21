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

import warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")

@dataclass
class StockOption:
    strike_price:    Union[float, np.ndarray]
    expiration_time: Union[float, np.ndarray]  # in years
    is_call:         bool

@dataclass
class CallStockOption(StockOption):
    def __init__(self, strike_price, expiration_time):
        super().__init__(strike_price, expiration_time, True)

@dataclass
class PutStockOption(StockOption):
    def __init__(self, strike_price, expiration_time):
        super().__init__(strike_price, expiration_time, False)

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

    # np.sqrt(np.var(data) / len(data))

    while length_conf_interval > absolute_error and iter_count < MAX_ITER:
        derivative_price_array = np.append(derivative_price_array, payoff(simulate(**args)['price']))
        iter_count+=1
        n+=batch_size
        length_conf_interval = C * np.sqrt(np.var(derivative_price_array) / derivative_price_array.size)

        if debug:
            print(f"Current price: {np.mean(derivative_price_array):.4f} +/- {get_len_conf_interval(derivative_price_array):.4f}")

    print(f"Number of iterations:   {iter_count}\nNumber of simulations:  {n}\n")

    return np.mean(derivative_price_array)

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
    vt[:]      = v0
    log_st     = np.zeros(n_simulations)
    log_st[:]  = np.log(s0)
    
    mean_St    = 0.
    mean_vt    = 0.
        
    vt[:]      = v0
    log_st[:]  = np.log(s0)
        
    V          = np.zeros([n_simulations, N_T])
    V[:, 0]    = vt

    logS       = np.zeros([n_simulations, N_T])
    logS[:, 0] = log_st

    Z          = np.random.normal(size=(n_simulations, N_T))
    Z_V = np.random.normal(size=(n_simulations, N_T))
    U = np.random.uniform(size=(n_simulations, N_T))
    # ksi = np.random.binomial(1, 1.0-p, size=(n_simulations, N_T))
    # eta = np.random.exponential(scale = 1., size=(n_simulations, N_T))



    for i in range(N_T - 1):
        m            = vbar+(V[:, i] - vbar)*E
        s_2          = (V[:, i]*(gamma**2)*E/kappa)*(1.0 - E) + (vbar*gamma**2)/(2.0*kappa)*((1-E)**2)
        Psi          = s_2/(m**2) # np.power

        cond         = np.where(Psi<=Psi_c)
        c            = 2 / Psi[cond]
        b            = c - 1 + np.sqrt(c*(c - 1.))
        a            = m[cond]/(1+b)
        b            = np.sqrt(b)
        # Z_V          = np.random.normal(size=cond[0].shape[0])
        V[cond, i+1] = a*((b+Z_V[cond, i])**2)

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
    vt[:]      = v0
    log_st     = np.zeros(n_simulations)
    log_st[:]  = np.log(s0)
    
    mean_St    = 0.
    mean_vt    = 0.
        
    vt[:]      = v0
    log_st[:]  = np.log(s0)
        
    V          = np.zeros([n_simulations, N_T])
    V[:, 0]    = vt

    logS       = np.zeros([n_simulations, N_T])
    logS[:, 0] = log_st

    Z          = np.random.normal(size=(n_simulations, N_T))
    Z_V        = np.random.normal(size=(n_simulations, N_T))
    
    
    dx = np.diff(r_x[0:2])[0]
    
    for i in range(N_T - 1):
        m            = vbar+(V[:, i] - vbar)*E
        s_2          = (V[:, i]*(gamma**2)*E/kappa)*(1.0 - E) + (vbar*gamma**2)/(2.0*kappa)*((1-E)**2)
        Psi          = s_2/(m**2) # np.power
        
        
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

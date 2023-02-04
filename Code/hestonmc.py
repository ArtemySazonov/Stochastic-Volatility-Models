import numpy as np
import math
sqrt = math.sqrt
exp  = math.exp
log  = math.log
from scipy.stats import norm

from typing import Union, Callable, Optional
from copy import error
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import newton, root_scalar

from numba import jit, njit, prange, float64
from numba.experimental import jitclass

if __name__ == '__main__':
    print("This is a module. Please import it.\n")
    exit(-1)

def european_call_payoff(maturity: float,
                         strike: float,
                         interest_rate: float = 0.):
    @jit
    def payoff(St: np.ndarray):
        DF = np.exp( - interest_rate * maturity)
        return np.maximum(St - strike, 0.)*DF

    return payoff
@jitclass([("kappa", float64),
           ("gamma", float64),
           ("rho", float64), 
           ("vbar", float64),
           ("v0", float64)])
class HestonParameters:
    def __init__(self, kappa, gamma, rho, vbar, v0):
        self.kappa = kappa
        self.gamma = gamma
        self.rho = rho
        self.vbar = vbar
        self.v0 = v0
        
@jitclass([("stock_price", float64),
           ("interest_rate", float64)])
class MarketState:
    def __init__(self, stock_price, interest_rate):
        self.stock_price = stock_price
        self.interest_rate = interest_rate

def get_len_conf_interval(data:             np.ndarray, 
                          confidence_level: float = 0.05):
    """Get the confidence interval length for a given confidence level.
    Args:
        data:             The data to compute the confidence interval for.
        confidence_level: The confidence level to use.
    
    Returns:
        The confidence interval.
    """
    return -2*norm.ppf(confidence_level*0.5) * sqrt(np.var(data) / len(data))

@njit
def set_seed(value):
    np.random.seed(value)

def mc_price(payoff:                 Callable,
             simulate:               Callable,
             state:                  MarketState,
             heston_params:          HestonParameters,
             T:                      float    = 1.,
             N_T:                    int      = 100,
             absolute_error:         float    = 0.01,
             confidence_level:       float    = 0.05,
             batch_size:             int      = 10_000,
             MAX_ITER:               int      = 100_000,
             control_variate_payoff: Callable = None,
             control_variate_iter:   int      = 1_000,
             verbose:                bool     = False,
             random_seed:            int      = None,
             **kwargs):
    """A function that performs a Monte-Carlo based pricing of a derivative with a given payoff (possibly path-dependent) under the Heston model.

    Args:
        payoff (Callable):                           Payoff function
        simulate (Callable):                         Simulation engine
        state (MarketState):                         Market state
        heston_params (HestonParameters):            Heston parameters
        T (float, optional):                         Contract expiration T. Defaults to 1.. 
        N_T (int, optional):                         Number of steps in time. Defaults to 100.
        absolute_error (float, optional):            Absolute error of the price. Defaults to 0.01 (corresponds to 1 cent). 
        confidence_level (float, optional):          Confidence level for the price. Defaults to 0.05.
        batch_size (int, optional):                  Path-batch size. Defaults to 10_000.
        MAX_ITER (int, optional):                    Maximum number of iterations. Defaults to 100_000.  
        control_variate_payoff (Callable, optional): Control variate payoff. Defaults to None.
        control_variate_iter (int, optional):        Number of iterations for the control variate. Defaults to 1_000.
        verbose (bool, optional):                    Verbose flag. If true, the technical information is printed. Defaults to False.
        random_seed (int, optional):                 Random seed. Defaults to None.
        **kwargs:                                    Additional arguments for the simulation engine.

    Returns:    
        The price(-s) of the derivative(-s).    
    """

    arg = {'state':         state,
           'heston_params': heston_params, 
           'T':             T, 
           'N_T':           N_T, 
           'n_simulations': batch_size}

    args       = {**arg, **kwargs}
    iter_count = 0   

    length_conf_interval = 1.
    n                    = 0
    C                    = -2*norm.ppf(confidence_level*0.5)
    sigma_n              = 0.
    batch_new            = np.zeros(batch_size, dtype=np.float64)
    current_Pt_sum       = 0.        

    if random_seed is not None:
        set_seed(random_seed)

    if control_variate_payoff is None:
        while length_conf_interval > absolute_error and iter_count < MAX_ITER:
            batch_new = payoff(simulate(**args)[0])
            iter_count+=1

            sigma_n = (sigma_n*(n-1.) + np.var(batch_new)*(2*batch_size - 1.))/(n + 2*batch_size - 1.)
            current_Pt_sum = current_Pt_sum + np.sum(batch_new) 

            n+=2*batch_size
            length_conf_interval = C * np.sqrt(sigma_n / n)
    else:
        S = simulate(control_variate_iter)
        c = np.cov(payoff(S), control_variate_payoff(S))
        theta = c[0, 1] / c[1, 1]
        while length_conf_interval > absolute_error and iter_count < MAX_ITER:
            batch_new = payoff(simulate(**args)[0]) - theta * control_variate_payoff(simulate(**args)[0])
            iter_count+=1

            sigma_n = (sigma_n*(n-1.) + np.var(batch_new)*(2*batch_size - 1.))/(n + 2*batch_size - 1.)
            current_Pt_sum = current_Pt_sum + np.sum(batch_new) 

            n+=2*batch_size
            length_conf_interval = C * np.sqrt(sigma_n / n)

    if verbose:
        if random_seed is not None:
            print(f"Random seed:                {random_seed}")

        if control_variate_payoff is not None:
            print(f"Control variate payoff:     {control_variate_payoff.__name__}")
            print(f"Control variate iterations: {control_variate_iter}")
        
        print(f"Number of simulate calls:   {iter_count}\nMAX_ITER:                   {MAX_ITER}\nNumber of paths:            {n}\nAbsolute error:             {absolute_error}\nLength of the conf intl:    {length_conf_interval}\nConfidence level:           {confidence_level}\n")

    return current_Pt_sum/n

@njit(parallel=True, cache=True, nogil=True)
def simulate_heston_euler(state:           MarketState,
                          heston_params:   HestonParameters,
                          T:               float = 1.,
                          N_T:             int   = 100,
                          n_simulations:   int   = 10_000
                          ) -> np.ndarray:
    """Simulation engine for the Heston model using the Euler scheme.

    Args:
        state (MarketState):              Market state.
        heston_params (HestonParameters): Parameters of the Heston model.
        T (float, optional):              Contract termination time expressed as a number of years. Defaults to 1..
        N_T (int, optional):              Number of steps in time. Defaults to 100.
        n_simulations (int, optional):    Number of simulations. Defaults to 10_000.

    Raises:
        error: Contract termination time must be positive.

    Returns:
        A tuple containing the simulated stock price and the simulated stochastic variance.
        The number of paths is doubled to account for the antithetic variates.
    """    
    if T <= 0:
        raise error("Contract termination time must be positive.")
    
    r, s0 = state.interest_rate, state.stock_price
    v0, rho, kappa, vbar, gamma = heston_params.v0, heston_params.rho, heston_params.kappa, heston_params.vbar, heston_params.gamma
    
    dt         = T/float(N_T)
    
    Z1         = np.random.standard_normal(size=(n_simulations, N_T))
    Z2         = np.random.standard_normal(size=(n_simulations, N_T))
    V          = np.zeros((2*n_simulations, N_T))
    V[:, 0]    = v0
    
    logS       = np.zeros((2*n_simulations, N_T))
    logS[:, 0] = np.log(s0)

    sqrt1_rho2 = sqrt(1-rho**2)

    for n in prange(n_simulations):
        for i in range(0,  N_T-1):
            vmax             = max(V[2*n, i],0)
            sqrtvmaxdt       = sqrt(vmax*dt)
            logS[2*n, i+1]   = logS[2*n, i] + (r - 0.5 * vmax) * dt + sqrtvmaxdt * Z1[n, i]
            V[2*n, i+1]      = V[2*n, i] + kappa*(vbar - vmax)*dt + gamma*sqrtvmaxdt*(rho*Z1[n, i]+sqrt1_rho2*Z2[n, i])

            vmax             = max(V[2*n+1, i],0)
            sqrtvmaxdt       = sqrt(vmax*dt)
            logS[2*n+1, i+1] = logS[2*n+1, i] + (r - 0.5 * vmax) * dt - sqrtvmaxdt * Z1[n, i]
            V[2*n+1, i+1]    = V[2*n+1, i] + kappa*(vbar - vmax)*dt - gamma*sqrtvmaxdt*(rho*Z1[n, i]+sqrt1_rho2*Z2[n, i])

    return [np.exp(logS[:, N_T-1]), V[:, N_T-1]]

@njit(parallel=True, cache=True, nogil=True)
def simulate_heston_andersen_qe(state:         MarketState,
                                heston_params: HestonParameters,
                                T:             float = 1.,
                                N_T:           int   = 100,
                                n_simulations: int   = 10_000,
                                Psi_c:         float = 1.5,
                                gamma_1:       float = 0.0     
                                ) -> np.ndarray: 
    """Simulation engine for the Heston model using the Quadratic-Exponential Andersen scheme.

    Args:
        state (MarketState):              Market state.
        heston_params (HestonParameters): Parameters of the Heston model.
        T (float, optional):              Contract termination time expressed as a non-integer amount of years. Defaults to 1..
        N_T (int, optional):              Number of steps in time. Defaults to 100.
        n_simulations (int, optional):    Number of simulations. Defaults to 10_000.
        Psi_c (float, optional):          Critical value of \psi, i.e. the moment of the scheme switching. Defaults to 1.5.
        gamma_1 (float, optional):        Integration parameter. Defaults to 0.5.

    Raises:
        Error: The critical value \psi_c must be in the interval [1,2]
        Error: The parameter \gamma_1 must be in the interval [0,1]

    Returns:
        A tuple containing the simulated stock price and the simulated stochastic variance.
        The number of paths is doubled to account for the antithetic variates.
    """    
    
    if Psi_c>2 or Psi_c<1:
        raise error('The critical value \psi_c must be in the interval [1,2]')
    if gamma_1 >1 or gamma_1<0:
        raise error('The parameter \gamma_1 must be in the interval [0,1]')
    if T <= 0:
        raise error("Contract termination time must be positive.")
        
    gamma_2 = 1.0 - gamma_1
    
    r, s0 = state.interest_rate, state.stock_price
    v0, rho, kappa, vbar, gamma = heston_params.v0, heston_params.rho, heston_params.kappa, heston_params.vbar, heston_params.gamma
    
    dt         = T/float(N_T)
    E          = exp(-kappa*dt)
    K_0        = -(rho*kappa*vbar/gamma)*dt
    K_1        = gamma_1 * dt * (rho*kappa/gamma - 0.5) - rho/gamma
    K_2        = gamma_2 * dt * (rho*kappa/gamma - 0.5) + rho/gamma
    K_3        = gamma_1 * dt * (1.0 - rho**2)
    K_4        = gamma_2 * dt * (1.0 - rho**2)
        
    V          = np.zeros((2*n_simulations, N_T))
    V[:, 0]    = v0

    logS       = np.zeros((2*n_simulations, N_T))
    logS[:, 0] = np.log(s0)

    Z          = np.random.standard_normal(size=(n_simulations, N_T))
    Z_V        = np.random.standard_normal(size=(n_simulations, N_T))    #do we need this?
    U          = np.random.random_sample(size=(n_simulations, N_T))   #do we need this?

    p1         = (1. - E)*(gamma**2)*E/kappa
    p2         = (vbar*gamma**2)/(2.0*kappa)*((1.-E)**2)
    p3         = vbar * (1.- E)
    rdtK0      = r*dt + K_0

    for n in prange(n_simulations):
        for i in range(N_T - 1):
            m   = p3 + V[2*n, i]*E
            s_2 = V[2*n, i]*p1 + p2
            Psi = s_2/(m**2) 

            if Psi <= Psi_c:
                c           = 2. / Psi
                b           = c - 1. + sqrt(c*(c - 1.))
                a           = m/(1.+b)
                b           = sqrt(b)
                V[2*n, i+1] = a*((b+Z_V[n, i])**2)
            else:
                p           = (Psi - 1)/(Psi + 1)
                beta        = (1.0 - p)/m
                V[2*n,i+1]  = 0. if U[n, i] < p else log((1-p)/(1-U[n, i]))/beta

            logS[2*n,i+1] = logS[2*n,i] + rdtK0 + K_1*V[2*n,i] + K_2*V[2*n,i+1] + sqrt(K_3*V[2*n,i]+K_4*V[2*n,i+1]) * Z[n,i]

            m   = p3 + V[2*n+1, i]*E
            s_2 = V[2*n+1, i]*p1 + p2
            Psi = s_2/(m**2) 

            if Psi <= Psi_c:
                c             = 2. / Psi
                b             = c - 1. + sqrt(c*(c - 1.))
                a             = m/(1.+b)
                b             = sqrt(b)
                V[2*n+1, i+1] = a*((b-Z_V[n, i])**2)
            else:
                p             = (Psi - 1)/(Psi + 1)
                beta          = (1.0 - p)/m
                V[2*n+1,i+1]  = 0. if 1-U[n, i] < p else log((1-p)/(U[n, i]))/beta

            logS[2*n+1,i+1] = logS[2*n+1,i] + rdtK0 + K_1*V[2*n+1,i] + K_2*V[2*n+1,i+1] - sqrt(K_3*V[2*n+1,i]+K_4*V[2*n+1,i+1]) * Z[n,i]
            
    return [np.exp(logS[:, N_T-1]), V[:, N_T-1]]

def calculate_r_for_andersen_tg(x_:      float,
                                maxiter: int = 2500, 
                                tol:     float = 1e-5
                                ):
    def foo(x: float):
        return x*norm.pdf(x) + norm.cdf(x)*(1+x**2) - (1+x_)*(norm.pdf(x) + x*norm.cdf(x))**2

    def foo_dif(x: float):
        return norm.pdf(x) - x**2 * norm.pdf(x) + norm.pdf(x)*(1+x**2) + 2*norm.cdf(x)*x - \
                2*(1+x_)*(norm.pdf(x) + x*norm.cdf(x))*(-norm.pdf(x)*x + norm.cdf(x) + x*norm.pdf(x) )

    def foo_dif2(x: float):
        return -x*norm.pdf(x) - 2*x* norm.pdf(x) + x**3 * norm.pdf(x) -x*norm.pdf(x)*(1+x**2) + \
                2*norm.cdf(x)*x + 2*norm.pdf(x)*x + 2*norm.cdf(x) + \
                2*(1+x_)*(-norm.pdf(x)*x + norm.cdf(x) + x*norm.pdf(x))**2 + \
                2*(1+x_)*(norm.pdf(x) + x*norm.cdf(x))*(x**2*norm.pdf(x) + norm.pdf(x) + norm.pdf(x) -x*norm.pdf(x) )

    return newton(foo,  x0 = 1/x_,fprime = foo_dif, fprime2 = foo_dif2, maxiter = maxiter , tol= tol )


@njit(parallel=True, cache=True, nogil=True)
def simulate_heston_andersen_tg(state:         MarketState,
                                heston_params: HestonParameters,
                                x_grid:        np.ndarray,
                                f_nu_grid:     np.ndarray,
                                f_sigma_grid:  np.ndarray,
                                T:             float = 1.,
                                N_T:           int   = 100,
                                n_simulations: int   = 10_000,
                                gamma_1:       float = 0.0
                                ) -> np.ndarray: 
    """ Simulation engine for the Heston model using the Truncated Gaussian Andersen scheme.

    Args:
        state (MarketState):              Market state.
        heston_params (HestonParameters): Parameters of the Heston model.
        x_grid (np.ndarray):              _description_
        f_nu_grid (np.ndarray):           _description_
        f_sigma_grid (np.ndarray):        _description_
        T (float, optional):              Contract termination time expressed as a non-integer amount of years. Defaults to 1..
        dt (float, optional):             Time step. Defaults to 1e-2.
        n_simulations (int, optional):    number of the simulations. Defaults to 10_000.
        gamma_1 (float, optional):        _description_. Defaults to 0.0.

    Raises:
        error: The parameter \gamma_1 must be in the interval [0,1].
        error: Contract termination time must be positive.

    Returns:
        A tuple containing the simulated stock price and the simulated stochastic variance.
        The number of paths is doubled to account for the antithetic variates.
    """    
    if gamma_1 >1 or gamma_1<0:
        raise error('The parameter \gamma_1 must be in the interval [0,1]')
    if T <= 0:
        raise error("Contract termination time must be positive.")
            
    r, s0 = state.interest_rate, state.stock_price
    v0, rho, kappa, vbar, gamma = heston_params.v0, heston_params.rho, heston_params.kappa, heston_params.vbar, heston_params.gamma
    
    gamma_2    = 1. - gamma_1
    dt         = T/float(N_T)
    E          = exp(-kappa*dt)
    K_0        = -(rho*kappa*vbar/gamma)*dt
    K_1        = gamma_1 * dt * (rho*kappa/gamma - 0.5) - rho/gamma
    K_2        = gamma_2 * dt * (rho*kappa/gamma - 0.5) + rho/gamma
    K_3        = gamma_1 * dt * (1.0 - rho**2)
    K_4        = gamma_2 * dt * (1.0 - rho**2)
        
    V          = np.zeros((2*n_simulations, N_T))
    V[:, 0]    = v0
    logS       = np.zeros((2*n_simulations, N_T))
    logS[:, 0] = np.log(s0)

    Z          = np.random.standard_normal(size=(n_simulations, N_T))
    Z_V        = np.random.standard_normal(size=(n_simulations, N_T))    #do we need this?
    U          = np.random.random_sample(size=(n_simulations, N_T))   #do we need this?

    p1         = (1. - E)*(gamma**2)*E/kappa
    p2         = (vbar*gamma**2)/(2.0*kappa)*((1.-E)**2)
    p3         = vbar * (1.- E)
    rdtK0      = r*dt + K_0
    dx         = x_grid[1] - x_grid[0]
    
    for n in prange(n_simulations):
        for i in range(N_T - 1):
            m               = p3 + V[2*n, i]*E
            s_2             = V[2*n, i]*p1 + p2
            Psi             = s_2/(m**2) 


            if Psi > x_grid[-1]:
                inx         = x_grid.shape[0] -1
            else:
                inx         = int(Psi/dx)
        
            nu              = m*f_nu_grid[inx]
            sigma           = sqrt(s_2)*f_sigma_grid[inx]

            V[2*n, i+1]     = max(nu + sigma*Z_V[n, i], 0)
            logS[2*n,i+1]   = logS[2*n,i] + rdtK0 + K_1*V[2*n,i] + K_2*V[2*n,i+1] + sqrt(K_3*V[2*n,i]+K_4*V[2*n,i+1]) * Z[n,i]

            m               = p3 + V[2*n+1, i]*E
            s_2             = V[2*n+1, i]*p1 + p2
            Psi             = s_2/(m**2) 

            if Psi > x_grid[-1]:
                inx         = x_grid.shape[0] -1
            else:
                inx             = int(Psi/dx)
        
            nu              = m * f_nu_grid[inx]
            sigma           = np.sqrt(s_2)*f_sigma_grid[inx]

            V[2*n+1,i+1]    = max(nu - sigma*Z_V[n, i], 0)
            logS[2*n+1,i+1] = logS[2*n+1,i] + rdtK0 + K_1*V[2*n+1,i] + K_2*V[2*n+1,i+1] - sqrt(K_3*V[2*n+1,i]+K_4*V[2*n+1,i+1]) * Z[n,i]
            
    return [np.exp(logS[:, N_T-1]), V[:, N_T-1]]























#non-working piece of code


def cir_chi_sq_sample(heston_params: HestonParameters,
                      dt:            float,
                      v_i:           np.ndarray,
                      n_simulations: int):
    """Samples chi_squared statistics for v_{i+1} conditional on 
       v_i and parameters of the Heston model. 
        
    Args:
        heston_params (HestonParameters): parameters of Heston model
        dt (float): T step 
        v_i: current volatility value
        n_simulations (int): number of simulations.
        
    Returns:
        np.ndarray: sampled chi_squared statistics 
    """
    kappa, vbar, gamma = heston_params.kappa, heston_params.vbar, heston_params.gamma
    
    barkappa=v_i*(4*kappa*np.exp(-kappa*dt))/(gamma**2 * (1 - np.exp(-kappa*dt)))
    d=(4*kappa*vbar)/(gamma**2)
    c=((gamma**2)/(4*kappa))*(1-np.exp(-kappa*dt))
    
    return  c*np.random.noncentral_chisquare(size = n_simulations, df   = d, nonc = barkappa)


def Phi(a:             Union[float, np.ndarray], 
        V:             Union[float, np.ndarray],
        T:             Union[float, np.ndarray],
        heston_params: HestonParameters
        ) -> np.ndarray:
    
    
    v0, rho, kappa, vbar, gamma = heston_params.v0, heston_params.rho, heston_params.kappa, \
                                        heston_params.vbar, heston_params.gamma
    dt = T[1::]-T[:-1:]
    
    A=np.ndarray(a)
    gamma_a = np.sqrt(kappa**2 - 2*gamma**2*1j*A).reshape(1,1,len(A)).T
    
    E1 = np.exp(-kappa*dt)
    E2 = np.exp(-gamma_a * dt)
        
    P1 = ((1.0-E1)*gamma_a/(kappa*(1.0-E2)))*np.exp(-0.5*(gamma_a - kappa)*dt)
    
    P2_2 = kappa * (1.0 + E1)/(1.0 - E1) - gamma_a*(1.0+E2)/(1-E2)
    P2 = np.exp( (V[:,1::]+V[:,:-1:])/(gamma_a**2) * P2_2 )
    
    P3_1 = np.sqrt(V[:,1::]*V[:,:-1:])*4*gamma_a * np.exp(-0.5*gamma_a*dt) /(gamma**2 * (1.0 - E2))
    P3_2 = np.sqrt(V[:,1::]*V[:,:-1:])*4*kappa*np.exp(-0.5*kappa*dt)/(gamma**2 * (1 - E1))
    d=(4*kappa*vbar)/(gamma**2)
    P3 = iv(0.5*d - 1, P3_1)/iv(0.5*d - 1, P3_2)
    
    return P1*P2*P3

def Pr(V:             np.ndarray, 
       T:          np.ndarray,
       X:             Union[np.ndarray, float],
       heston_params: HestonParameters,
       h:             float=1e-2, 
       eps:           float=1e-2
       ) -> np.ndarray:
    
    x=np.ndarray(X)
    P=h*x/np.pi
    S = 0.0
    j = 1
    while(True):
        Sin=np.sin(h*j*x)/j
        Phi_hj=Phi(h*j, V, T, heston_params)
        S+=Sin.reshape(1,1,len(x)).T * Phi_hj[0]
        if np.all(Phi_hj[0]<np.pi*eps*j/2.0):
            break
        j=j+1
    
    S=S*2.0/np.pi
    return P+S

def IV(V:             np.ndarray, 
       T:          np.ndarray,
       heston_params: HestonParameters,
       h:             float=1e-2, 
       eps:           float=1e-2
       ) -> np.ndarray:
    
    U=np.random.uniform(size=(V.shape[0], V.shape[1] - 1))
    
    def f(x,i,j):
        P=Pr(V, T, x, heston_params, h, eps)
        return (P-U)[0][i,j]
    
    IVar = np.zeros((V.shape[0], V.shape[1] - 1))
    
    for i in range(IVar.shape[0]):
        for j in range(IVar.shape[1]):     
            IVar[i,j]=root_scalar(f, args=(i,j), x0=0.5, method='newton')
    return IVar
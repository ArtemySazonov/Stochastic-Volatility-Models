import numpy as np
from numba import jit, njit, prange, float64
from numba.experimental import jitclass

if __name__ == '__main__':
    print("This is a module. Please import it.\n")
    exit(-1)

def european_call_payoff(maturity: float,
                         strike: float,
                         interest_rate: float = 0.):
    @njit
    def european_call(S: np.ndarray):
        DF = np.exp( - interest_rate * maturity)
        return np.maximum(S[:, -1] - strike, 0.)*DF

    return european_call

def european_put_payoff(maturity: float,
                        strike: float,
                        interest_rate: float = 0.):
    @njit
    def european_put(S: np.ndarray):
        DF = np.exp( - interest_rate * maturity)
        return np.maximum(strike - S[:, -1], 0.)*DF

    return european_put

def asian_call_AM_payoff(maturity: float,
                         strike: float,
                         interest_rate: float = 0.):
    @njit
    def asian_call_AM(S: np.ndarray):
        dt = maturity/np.shape(S)[1]
        I = np.sum(S, axis=1) * dt
        DF = np.exp( - interest_rate * maturity)
        return np.maximum(I - strike, 0.)*DF

    return asian_call_AM

def asian_put_AM_payoff(maturity: float,
                        strike: float,
                        interest_rate: float = 0.):
    @njit
    def asian_put_AM(S: np.ndarray):
        dt = maturity/np.shape(S)[1]
        I = np.sum(S, axis=1) * dt
        DF = np.exp( - interest_rate * maturity)
        return np.maximum(strike - I, 0.)*DF

    return asian_put_AM

def asian_call_GM_payoff(maturity: float,
                         strike: float,
                         interest_rate: float = 0.):
    @njit
    def asian_call_GM(S: np.ndarray):
        dt = maturity/np.shape(S)[1]
        I = np.exp(np.sum(np.log(S), axis=1) * dt)
        DF = np.exp( - interest_rate * maturity)
        return np.maximum(I - strike, 0.)*DF

    return asian_call_GM

def asian_put_GM_payoff(maturity: float,
                        strike: float,
                        interest_rate: float = 0.):
    @njit
    def asian_put_GM(S: np.ndarray):
        dt = maturity/np.shape(S)[1]
        I = np.exp(np.sum(np.log(S), axis=1) * dt)
        DF = np.exp( - interest_rate * maturity)
        return np.maximum(strike - I, 0.)*DF

    return asian_put_GM
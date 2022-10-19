# Student Research Group "Stochastic Volatility Models"
## Project 'Heston-2'
**Tasks**
1. Report: a review of Heston numerical simulations

* Give a talk on Euler, Broadie-Kaya, and Andersen schemes 

* Price the options using MC simulations. Compare the MC prices with closed-form theoretical prices for the vanilla European call options

* Calculate the greeks numerically

2. A numerical experiment and a report: improve a library implementation

* Implement E+M method from the article by Mrázek, Pospíšil (2017). Use Numba and/or CUDA if possible

* Reimplement Broadie-Kaya method to speed it up

3. A numerical experiment: compare different schemes of Heston simulation

* For a given class of derivatives (both vanilla and exotics) compare the performance

* Write a report


**Bibliography** 
1. Heston (1993), "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options”

2. Gatheral (2006), "Volatility Surface", Ch. 2

3. Andersen (2006), “Efficient Simulation of the Heston Stochastic Volatility Model”

4. Broadie, Kaya (2006), “Exact Simulation of Stochastic Volatility and Other Affine Jump Diffusion Processes”

5. Mrázek, Pospíšil (2017), “Calibration and simulation of Heston model”

6. Rouah (2013), “The Heston Model and Its Extensions in Matlab and C#”, Ch. 3

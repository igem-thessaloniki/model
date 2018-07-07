from scipy.integrate import odeint
import numpy as np

N = 1000

K = np.random.normal(0.001, 0.05*0.001, N)
CA0 = np.random.normal(1, 0.05*1, N)

X = [] # to store answer in
for k, Ca0 in zip(K, CA0):
    # define ODE
    def ode(X, t):
        ra = -k * (Ca0 * (1 - X))**2
        return -ra / Ca0

    X0 = 0
    tspan = np.linspace(0,3600)

    sol = odeint(ode, X0, tspan)

    X += [sol[-1][0]]

s = 'Final conversion at one hour is {0:1.3f} +- {1:1.3f} (1 sigma)'
print( s.format(np.average(X),
               np.std(X)))
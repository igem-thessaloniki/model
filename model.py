import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# copy number
c = 5.0
# cooperativity of repressor binding
n = 2.0
# transcription rates
aR = 1.0
aGmax = 1.0
aGmin = 1.0
# degradation rates
yR = 1.0
yG = 1.0
yM = 1.0
# translation rates
bR = 1.0
bG = 1.0
# on and off rates of repressor binding to the promoter
kRon = 1.0
kRoff = 0.5

def model(z, t):
    mR = z[0]
    R = z[1]
    PG = z[2]
    PGR = z[3]
    mG = z[4]
    G = z[5]
    dmRdt = c * aR - yM * mR
    dRdt = bR * mR - yR * R - n * kRon * (R ** n) * PG + n * kRoff * PGR + (n - 1) * n * yR * PGR
    dPGdt = kRoff * PGR - kRon * (R ** n) * PG + n * yR * PGR
    dPGRdt = kRon * (R ** n) * PG - kRoff * PGR - n * yR * PGR
    dmGdt = aGmax * PG + aGmin * PGR - yM * mG
    dGdt = bG * mG - yG * G
    dzdt = [dmRdt, dRdt, dPGdt, dPGRdt, dmGdt, dGdt]
    return dzdt

z0 = [0, 0, 0, 0, 0, 0]

t = np.linspace(0, 10, 1000)

z = odeint(model, z0, t)
plt.plot(t,z[:,0],'b-',label='mR')
plt.plot(t,z[:,1],'r-',label='R')
plt.plot(t,z[:,2],'b-.',label='PG')
# plt.plot(t,z[:,3],'r-.',label='PGR')
# plt.plot(t,z[:,4],'b--',label='mG')
# plt.plot(t,z[:,5],'g-',label='G')
plt.ylabel('response')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()
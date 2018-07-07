import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# cooperativity of repressor binding
n = 1.0

# Samples
N = 100

# transcription rates
aR = np.random.normal(3.0, 0.5, N)
aGmax = np.random.normal(1.0, 0.5, N)
aGmin = np.random.normal(0.0001, 0.0001, N)

# degradation rates
yR = np.random.normal(0.2, 0.1, N)
yG = np.random.normal(0.8, 0.1, N)
yM = np.random.normal(0.5, 0.1, N)

# translation rates
bR = np.random.normal(3.0, 0.5, N)
bG = np.random.normal(1.0, 0.5, N)

# on and off rates of repressor binding to the promoter
kRon = 0.9
kRoff = 4000
kDn = kRoff / kRon

def wrap(c, aR, aGmax, aGmin, yR, yG, yM, bR, bG):
    results = []
    for aR, aGmax, aGmin, yR, yG, yM, bR, bG in zip(aR, aGmax, aGmin, yR, yG, yM, bR, bG):
        def model(z, t):
            mR = z[0]
            R = z[1]
            PG = z[2]
            PGR = z[3]
            mG = z[4]
            G = z[5]
            Rn = R

            dmRdt = c * aR - yM * mR
            # dRdt = bR() * mR - yR * R - n * kRon * Rn * PG + n * kRoff * PGR + (n - 1) * n * yR * PGR
            dRdt = bR * mR - yR * R
            dPGdt = kRoff * PGR - kRon * Rn * PG + n * yR * PGR
            dPGRdt = kRon * Rn * PG - kRoff * PGR - n * yR * PGR
            dmGdt = aGmax * PG + aGmin * PGR - yM * mG
            # dmGdt = c * (aGmin + (aGmax - aGmin) * (kDn / (kDn + Rn))) - yM * mG
            # dmGdt = c * (aGmax * kDn / Rn) - yM * mG
            dGdt = bG * mG - yG * G
            dzdt = [dmRdt, dRdt, dPGdt, dPGRdt, dmGdt, dGdt]
            return dzdt

        z0 = [0, 0, c, 0, 0, 0]
        t = np.linspace(0, 20, 1000)
        result = odeint(model, z0, t)
        results.append(result)
    return results


final = False
for copy_n in range(100, 200, 10):
    t = np.linspace(0, 20, 1000)
    z = wrap(copy_n, aR, aGmax, aGmin, yR, yG, yM, bR, bG)
    final = z[0]
    for i in range(1, N):
        final += z[i]
    final /= N
    print(copy_n,final[:, 5][-1])

t = np.linspace(0, 20, 1000)
# plt.plot(t,final[:,0],'b-',label='mR')
# plt.plot(t,final[:,1],'r-',label='R')
plt.plot(t,final[:,2],'b-.',label='PG')
plt.plot(t,final[:,3],'r-.',label='PGR')
plt.plot(t,final[:,4],'b--',label='mG')
plt.plot(t,final[:,5],'g-',label='G')
plt.ylabel('concentration')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()
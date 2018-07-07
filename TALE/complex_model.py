import numpy as np
import uncertainties as u
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# cooperativity of repressor binding
n = 1.0
# transcription rates
aR = u.ufloat(2.0, 0.05)
aGmax = u.ufloat(2.0, 0.05)
aGmin = u.ufloat(2.0, 0.05)

# degradation rates
yR = aGmin = u.ufloat(0.25, 0.05)
yG = aGmin = u.ufloat(0.25, 0.05)
yM = aGmin = u.ufloat(0.25, 0.05)

# translation rates
bR = u.ufloat(4.0, 0.05)
bG = u.ufloat(4.0, 0.05)
print(bR.n)

# on and off rates of repressor binding to the promoter
kRon = 0.9
kRoff = 4000
kDn = kRoff / kRon

def wrap(c, yR, yM, aR, aGmax, aGmin, bG, yG, bR):
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
    return odeint(model, z0, t)


z = []
for i in range(100, 200, 10):
    z = wrap(i, yR, yM, aR, aGmax, aGmin, bG, yG, bR)
    print(i,z[:, 5][-1])

# plt.plot(t,z[:,0],'b-',label='mR')
# plt.plot(t,z[:,1],'r-',label='R')
# plt.plot(t,z[:,2],'b-.',label='PG')
# plt.plot(t,z[:,3],'r-.',label='PGR')
# plt.plot(t,z[:,4],'b--',label='mG')
# plt.plot(t,z[:,5],'g-',label='G')
plt.ylabel('concentration')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()
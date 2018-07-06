import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# copy number
c = 0
# cooperativity of repressor binding
n = 1.0
# transcription rates
amRcas9 = 1.0
asgRNA = 1.0
aGmax = 1.0
aGmin = 0.0001
# degradation rates
ysgRNA = 0.1
ymRcas9 = 0.2
ycas9 = 0.2
yR = 0.1
ymRG = 0.2
yG = 0.2
# translation rates
bcas9 = 4.0
bG = 1.0
# on and off rates of repressor binding to the promoter
g = 1.0
kBindOn = 1.0
kBindOff = 0.000001
kRon = 1
kRoff = 0.1
# kDn = kRoff / kRon

def model(z, t):
    mRcas9 = z[0]
    cas9 = z[1]
    sgRNA = z[2]
    R = z[3]
    PG = z[4]
    PGR = z[5]
    mG = z[6]
    G = z[7]
    Rn = R
    dmRcas9dt = c * amRcas9 - ymRcas9 * mRcas9
    dcas9dt = mRcas9 * bcas9 - ycas9 * cas9 - kBindOn * cas9 * sgRNA + kBindOff * R
    dsgRNAdt = c * asgRNA - ysgRNA * sgRNA - kBindOn * cas9 * sgRNA + kBindOff * R
    dRdt = kBindOn * cas9 * sgRNA - yR * R - kBindOff * R
    dPGdt = kRoff * PGR - kRon * Rn * PG + n * yR * PGR
    dPGRdt = kRon * Rn * PG - kRoff * PGR - n * yR * PGR
    dmGdt = aGmax * PG + aGmin * PGR - ymRG * mG
    dGdt = bG * mG - yG * G
    dzdt = [dmRcas9dt, dcas9dt, dsgRNAdt, dRdt, dPGdt, dPGRdt, dmGdt, dGdt]
    return dzdt

z = []
for i in range(100, 200, 10):
    c = i
    t = np.linspace(0, 20, 1000)
    z0 = [0, 0, 0, 0, c, 0, 0, 0]
    z = odeint(model, z0, t)
    print(c,z[:, 7][-1])

# plt.plot(t,z[:,0],'b-',label='mRcas9')
# plt.plot(t,z[:,1],'r-',label='cas9')
# plt.plot(t,z[:,2],'g-',label='sgRNA')
# plt.plot(t,z[:,3],'b-.',label='R')
plt.plot(t,z[:,4],'r-.',label='PG')
plt.plot(t,z[:,5],'g-.',label='PGR')
# plt.plot(t,z[:,5],'b--',label='mG')
# plt.plot(t,z[:,5],'r--',label='G')
plt.ylabel('concentration')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()
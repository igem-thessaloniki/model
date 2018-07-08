import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# copy number
c = 0
# cooperativity of repressor binding
n = 1.0
# transcription rates
aR = 3.0
aGmax = 1.0
aGmin = 0.0001
# degradation rates
yR = 0.2
yG = 0.8
yM = 0.5
# translation rates
bR = 3.0
bG = 1.0
# on and off rates of repressor binding to the promoter
kRon = 0.9
kRoff = 4000
kDn = kRoff / kRon

def wrap(c, aR, aGmax, aGmin, yR, yG, yM, bR, bG):
    def model(z, t):
        mR = z[0]
        R = z[1]
        PG = z[2]
        PGR = z[3]
        mG = z[4]
        G = z[5]
        Rn = R
        dmRdt = c * aR - yM * mR
        # dRdt = bR * mR - yR * R - n * kRon * Rn * PG + n * kRoff * PGR + (n - 1) * n * yR * PGR
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

c_number_min = 5
c_number_max = 300
c_number_step = 5

def main():
    final_results = []
    for c in range(c_number_min, c_number_max, c_number_step):
        z = wrap(c, aR, aGmax, aGmin, yR, yG, yM, bR, bG)
        final_results.append(z[:,5][-1])
        if False:
            t = np.linspace(0, 20, 1000)
            # plt.plot(t,z[:,0],'b-',label='mR')
            # plt.plot(t,z[:,1],'r-',label='R')
            plt.plot(t,z[:,2],'b-.',label='PG')
            plt.plot(t,z[:,3],'r-.',label='PGR')
            # plt.plot(t,z[:,4],'b--',label='mG')
            # plt.plot(t,z[:,5],'g-',label='G')
            plt.ylabel('concentration')
            plt.xlabel('time')
            plt.legend(loc='best')
            plt.show()
    if False:
        plt.plot(range(c_number_min, c_number_max, c_number_step),final_results,'g-',label='Model')
        plt.plot(range(c_number_min, c_number_max, c_number_step),range(c_number_min, c_number_max, c_number_step),'r-.',label='Reality')
        plt.ylabel('GOI')
        plt.xlabel('Copy Number')
        plt.legend(loc='best')
        plt.show()

    beta = bR * aR / (yR * yM)
    error = kDn / beta * c_number_min
    strength = kDn / (kDn + beta * c_number_min)
    return {
        'beta': beta,
        'error': error,
        'strength': strength,
        'min_goi': min(final_results),
        'max_goi': max(final_results),
    }

if __name__ == '__main__':
    r = main()
    print(r)
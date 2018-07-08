import numpy as np
np.random.seed(0)

NO_OF_PARAMETERS=2

#checking if parameters are out of outofbounds
def outofbounds(index,value):
    print(index)
    print(value)
    if index==0:
        if value<=5 and value>=0:
            return False
    if index==1:
        if value<=6 and value>=1:
            return False
    return True

def calcDev(w):
    return (50*w[0]- 20*w[1])

# the function we want to optimize
def f(w):
    dev=calcDev(w)
    reward = -(np.square(devgoal - dev))
    return reward

# hyperparameters
npop = 50 # population size
sigma = 0.1 # noise standard deviation
alpha = 0.001 # learning rate

# start the optimization
devgoal=0.5 #THE GOAL IS THAT THE DIFFERENCE BETWEEN LOWEST AND HIGHEST GOI IS <=DEVGOAL
#w=np.array(NO_OF_PARAMETERS)
w=[0,0]
for i in range (NO_OF_PARAMETERS):  #i= 0, 1 | 0=kon, 1=koff
    while (True):
        w[i]=np.random.randn() # our initial guess is random
        if outofbounds(i,w[i])==False:
            break


i=0
while i<=300 and calcDev(w)>devgoal:
  # print current fitness of the most likely parameter setting
  if i % 20 == 0:
    print('iter %d. w: %s, solution: %s, reward: %f' %
          (i, str(w), str(calcDev(w)), f(w)))

  # initialize memory for a population of w's, and their rewards
  N = np.random.randn(npop, NO_OF_PARAMETERS) # samples from a normal distribution N(0,1)
  R = np.zeros(npop)
  for j in range(npop):
    w_try = w + sigma*N[j] # jitter w using gaussian of sigma 0.1
    R[j] = f(w_try) # evaluate the jittered version

  # standardize the rewards to have a gaussian distribution
  A = (R - np.mean(R)) / np.std(R)
  # perform the parameter update. The matrix multiply below
  # is just an efficient way to sum up all the rows of the noise matrix N,
  # where each row N[j] is weighted by A[j]
  w = w + alpha/(npop*sigma) * np.dot(N.T, A)

  i+=1

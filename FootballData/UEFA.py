import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# specify forward model
log5 = np.log(5.0)
Nteams = 10
A = np.linspace(1.0, 0.4, Nteams)
D = A
AD = np.concatenate((A, D))
beta = np.array(AD).transpose()*log5

# TODO: add home team advantage beta

M = 2
K = M*Nteams*(Nteams-1)

X = np.zeros((2*Nteams, 2*K))
k = 0
for i in range(0, Nteams):
    for j in range(i+1, Nteams):
        for m in range(0, M):
            # first game
            X[i, k] = 1
            X[j+Nteams, k] = -1
            k = k + 1
            # second game
            X[j, k] = 1
            X[i+Nteams, k] = -1
            k = k + 1

# TODO: add home team advantage line in X

print(X)
print(beta)

lam = np.exp(beta.transpose().dot(X))

y = np.random.poisson(lam)
#print(lam)
print(y)


gamma_model = sm.GLM(y.transpose(), X.transpose(), family=sm.families.Poisson())
gamma_results = gamma_model.fit()
print(gamma_results.summary())

betahat = gamma_results.params
yhat = gamma_results.fittedvalues

# shift
offset = beta[0]-betahat[0]
betahat = betahat + offset
print(beta-betahat)

#plt.scatter(beta, betahat)
#plt.show()

plt.scatter(y, yhat)
plt.show()
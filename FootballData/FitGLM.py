import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import myFunctions

fixturesIn = myFunctions.getFixtures(431)
fixturesLight = myFunctions.getFixturesLight(fixturesIn)
allTeams = myFunctions.getTeams(fixturesIn)

# Theta = [A_1,A2,...,A_T,D_1,D_2,...,D_T] will be the unknown strengths
# F will be the fixtures matrix, i.e. the "design matrix"
# F "loads" on the attack and defense strengths of the fixture's teams and
# delta is the observations
# delta holds the measured strength difference \hat{(A_i-D_j)}

Nteams = len(allTeams)
K = len(fixturesLight)
X = np.zeros(shape=(2 * Nteams, 2 * K))
y = np.zeros(shape=(1, 2 * K))

k = 0
for fixtureLight in fixturesLight.values():
    i = allTeams[fixtureLight[0]][0]  # home team
    j = allTeams[fixtureLight[1]][0]  # away team
    # first col: home team perspective
    X[i, k] = 1
    X[j + Nteams, k] = -1
    y[0, k] = fixtureLight[2]
    k += 1

    # second col: away team perspective
    X[j, k] = 1
    X[i + Nteams, k] = -1
    y[0, k] = fixtureLight[3]
    k += 1

gamma_model = sm.GLM(y.transpose(), X.transpose(), family=sm.families.Poisson())
gamma_results = gamma_model.fit()

betahat = gamma_results.params

A_hat = betahat[0:Nteams]
D_hat = betahat[Nteams:]
teamIds = list(allTeams.keys())
for hisId in teamIds:
    print('{}, A = {:.1f}, D = {:.1f}'.format(allTeams[hisId][1], A_hat[allTeams[hisId][0]], D_hat[allTeams[hisId][0]]))

print('done')

yhat = gamma_results.fittedvalues
plt.scatter(y, yhat)
plt.show()
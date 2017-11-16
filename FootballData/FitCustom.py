import numpy as np
import myFunctions

myg = myFunctions.g
myginv = myFunctions.ginv
a = 6; b = 5; c = 2/3

allTeams = {0: [0, 'BVB'], 1: [1, 'Werder'], 2: [2, 'HSV'], 3: [3, 'Bayern']}
Attack = [5, 3, 1, 4]
Defend = [1, 2, 3, 4]

allTeams = {0: [0, 'A'], 1: [1, 'B'], 2: [2, 'C'], 3: [3, 'D'], 4: [4, 'E'], 5: [5, 'F'], 6: [6, 'G'], 7: [7, 'H']}
Attack = [4, 4, 3, 3, 2, 2, 1, 1]
Defend = [4, 3, 2, 1, 4, 3, 2, 1]

cnt = 0
fixturesLight = dict()
M = 5
for i in range(0,len(allTeams)-1):
    for j in range(i+1,len(allTeams)):
        for rep in range(0,M):
            cnt+=1
            lai = myg(Attack[i],Defend[j],a,b,c)
            ni = np.random.poisson(lai)
            laj = myg(Attack[j],Defend[i],a,b,c)
            nj = np.random.poisson(laj)
            fixturesLight[cnt] = [i, j, ni, nj]

# fixturesIn = myFunctions.getFixtures()
# fixturesLight = myFunctions.getFixturesLight(fixturesIn)
# allTeams = myFunctions.getTeams(fixturesIn)

# Theta = [A_1,A2,...,A_T,D_1,D_2,...,D_T] will be the unknown strengths
# F will be the fixtures matrix, i.e. the "design matrix"
# F "loads" on the attack and defense strengths of the fixture's teams and
# delta is the observations
# delta holds the measured strength difference \hat{(A_i-D_j)}
alpha_0 = 0.01
beta_0 = 0.01

T = len(allTeams)
K = len(fixturesLight)
F = np.zeros(shape=(2*K+1, 2*T))
delta = np.zeros(shape=(2*K+1, 1))
w = np.ones(shape=(2*K+1))


# make a GLM model here: first read wiki, then run dummy, then read my notes on this project, run it again

k = 0
for fixtureLight in fixturesLight.values():
    # thereby represents the left-hand side of the equation as A_i-D_j
    # first row: home team perspective
    F[2*k,allTeams[fixtureLight[0]][0]] = 1
    F[2*k,T+allTeams[fixtureLight[1]][0]] = -1
    # second row: away team perspective
    F[2*k+1,allTeams[fixtureLight[1]][0]] = 1
    F[2*k+1,T+allTeams[fixtureLight[0]][0]] = -1

    # first row: home team perspective
    la_hat = fixtureLight[2] # MLE of lambda = n of home team
    w[2*k] = 1.0/((alpha_0+la_hat)/(beta_0+1)**2)
    delta[2*k] = myginv(la_hat,a,b,c)
    # second row: away team perspective
    la_hat = fixtureLight[3] # MLE of lambda = n of away team
    delta[2*k+1] = myginv(la_hat,a,b,c)
    w[2*k+1] = 1.0/((alpha_0+la_hat)/(beta_0+1)**2)

    # increment counter
    k+=1

F[-1,1] = 1
delta[-1] = 5
w[-1] = 1
W = np.diag(w)

Theta_hat2 = myFunctions.LS3(F,delta,W)

Attack_hat = Theta_hat2[0:T]
Defend_hat = Theta_hat2[T:]
teamIds = list(allTeams.keys())
for i in range(0,len(teamIds)):
    teamId = teamIds[i]
    # print('{:%s}, A = {:%.1f}, D = {:%.1f}'.format(allTeams[teamId][1],Attack[i],Defend[i]))
    # print(teamId) # external id = key
    # print(allTeams[teamId][0]) # internal id = index in strength vector
    print(allTeams[teamId][1])
    print(Attack_hat[i])
    print(Defend_hat[i])

print('done')


# for teamKey in allTeams:
#     print(teamKey,': ',allTeams[teamKey])

# for fixtureKey in fixturesLight:
#     print(fixtureKey,': ',fixturesLight[fixtureKey])


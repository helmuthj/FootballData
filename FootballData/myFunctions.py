import http.client
import json
import numpy as np
import math
import pathlib

# '/v1/competitions/430/fixtures'  #  BuLi 16/17

def requestCompetition(idCompetition):
    return '/v1/competitions/{}/fixtures'.format(idCompetition)


def fileCompetition(idCompetition):
    return './Data/Competition_{}.json'.format(idCompetition)


def getFixtures(idCompetition, reload=False):
    p = pathlib.Path(fileCompetition(idCompetition))
    if reload:
        fixtures = downloadFixtures(requestCompetition(idCompetition))
        with p.open(mode='w', encoding='utf8') as outfile:
            json.dump(fixtures, outfile, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        return fixtures

    else:
        try:
            with p.open(mode='r') as infile:
                fixtures = json.load(infile)
                return fixtures
        except OSError:
            fixtures = downloadFixtures(requestCompetition(idCompetition))
            with p.open(mode='w', encoding='utf8') as outfile:
                json.dump(fixtures, outfile, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
            return fixtures


def downloadFixtures(leagueString):
    connection = http.client.HTTPConnection('api.football-data.org')
    headers = {'X-Auth-Token': 'd8af2d37e3ad4ea6b36242a73be211f3', 'X-Response-Control': 'minified'}
    request = leagueString
    connection.request('GET', request, None, headers)
    response = connection.getresponse()
    foo = json.loads(response.read().decode())
    fixtures = foo['fixtures']
    # print(json.dumps(response4, indent=4, sort_keys=True))
    return fixtures


def getFixturesLight(fixturesIn):
    fixturesLight = dict()
    for fixtureIn in fixturesIn:
        if fixtureIn['status']=='FINISHED':
            result = fixtureIn['result']
            fixtureLight = [fixtureIn['homeTeamId'], fixtureIn['awayTeamId'], result['goalsHomeTeam'], result['goalsAwayTeam']]
            fixturesLight[fixtureIn['id']] = fixtureLight
    return fixturesLight


def getTeams(fixturesIn):
    allTeams = dict()  # map from internal id (running number) to football-data.org's internal team code and team name
    T = 0  # Number of teams
    for fixtureIn in fixturesIn:
        if fixtureIn['homeTeamId'] not in allTeams.keys():
            allTeams[fixtureIn['homeTeamId']] = [T, fixtureIn['homeTeamName']]
            T += 1

        if fixtureIn['awayTeamId'] not in allTeams.keys():
            allTeams[fixtureIn['awayTeamId']] = [T, fixtureIn['awayTeamName']]
            T += 1
    return allTeams


def LS1(A,b):
    # Solves A*x=b in a least squares sense
    x = np.linalg.lstsq(A,b)
    return x


def LS2(A,b):
    # Solves A*x=b in a least squares sense
    AA = A.transpose().dot(A)
    invAA = np.linalg.inv(AA)
    invAAA = invAA.dot(A.transpose())
    x = invAAA.dot(b)
    return x

def LS3(A,b,W):
    # Solves A*x=b in a weighted least squares sense
    AA = A.transpose().dot(W).dot(A) # A^T*W*A
    invAA = np.linalg.inv(AA)
    x = invAA.dot(A.transpose()).dot(W).dot(b)
    return x

def g(A,D,a,b,c):
    la = a/(b*math.exp(-c*(A-D))+1)
    return la


def ginv(la,a,b,c):
    if la>0 and la<a:
        AlessD = -math.log(a/(la*b)-1/b)/c
    if la<=0:
        AlessD = -10
    if la>=a:
        AlessD = 10
    return AlessD


def g2(A,D,a,b,c):
    AlessD = A-D
    if AlessD <= -2.0:
        la = 0
    if AlessD > -2.0 and AlessD < 0:
        la = 0.5
    if AlessD >= 0 and AlessD < a:
        la = AlessD+1
    if AlessD >= a:
        la = a
    return la


def ginv2(la, a, b, c):
    if la <= 0:
        AlessD = -2
    if la >= 0 and la < 1:
        AlessD = 2*la-2
    if la >= 1 and la < a:
        AlessD = la-1
    if la >= a:
        AlessD = a
    return AlessD
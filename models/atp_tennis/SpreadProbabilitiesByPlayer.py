import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import statsmodels.formula.api as smf
from models.atp_tennis.SpreadMonteCarlo import abs_probabilities_per_surface


np.random.seed(1)

conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
slam_wins = pd.read_sql('select * from atp_matches_spread_probabilities_slam_wins', conn)
slam_losses = pd.read_sql('select * from atp_matches_spread_probabilities_slam_losses', conn)
wins = pd.read_sql('select * from atp_matches_spread_probabilities_win', conn)
losses = pd.read_sql('select * from atp_matches_spread_probabilities_losses', conn)


def build_cumulative_probabilities(probabilities):
    probabilities_over = {}
    for k in probabilities:
        probabilities_over[k] = 0.
        for j in probabilities:
            if j > k:
                probabilities_over[k] += probabilities[j]
    return probabilities_over


def totals_prob(player, tournament, year, total, is_grand_slam, priors_per_surface, surface='Hard', win=True,
                alpha=5.0):
    if is_grand_slam:
        r = range(-18, 19, 1)
        if win:
            prior = priors_per_surface[surface][2]
            sql = slam_wins
        else:
            prior = priors_per_surface[surface][3]
            sql = slam_losses
    else:
        r = range(-12, 13, 1)
        if win:
            prior = priors_per_surface[surface][0]
            sql = wins
        else:
            prior = priors_per_surface[surface][1]
            sql = losses

    row = sql[((sql.player_id == player) & (sql.tournament == tournament) & (sql.year == year))]
    probabilities = prior.copy()
    if row.shape[0] > 0:
        for k in probabilities:
            probabilities[k] *= alpha * 100.0
        for i in r:
            if i < 0:
                x = int(row['minus' + str(abs(i))])
            elif i > 0:
                x = int(row['plus' + str(i)])
            else:
                x = int(row['even'])
            probabilities[i] += x

        s = 0.0
        for _, v in probabilities.items():
            s += v

        if s > 0:
            for k in probabilities:
                probabilities[k] /= s

    probabilities_over = build_cumulative_probabilities(probabilities)
    return probabilities_over[-int(spread)]


def spread_prob(player, tournament, year, spread, is_grand_slam, priors_per_surface, surface='Hard', win=True, alpha=5.0):
    if is_grand_slam:
        r = range(-18, 19, 1)
        if win:
            prior = priors_per_surface[surface][2]
            sql = slam_wins
        else:
            prior = priors_per_surface[surface][3]
            sql = slam_losses
    else:
        r = range(-12, 13, 1)
        if win:
            prior = priors_per_surface[surface][0]
            sql = wins
        else:
            prior = priors_per_surface[surface][1]
            sql = losses

    row = sql[((sql.player_id == player) & (sql.tournament == tournament) & (sql.year == year))]
    probabilities = prior.copy()
    if row.shape[0] > 0:
        for k in probabilities:
            probabilities[k] *= alpha * 100.0
        for i in r:
            if i < 0:
                x = int(row['minus'+str(abs(i))])
            elif i > 0:
                x = int(row['plus' + str(i)])
            else:
                x = int(row['even'])
            probabilities[i] += x

        s = 0.0
        for _, v in probabilities.items():
            s += v

        if s > 0:
            for k in probabilities:
                probabilities[k] /= s

    probabilities_over = build_cumulative_probabilities(probabilities)
    return probabilities_over[-int(spread)]


print(spread_prob('roger-federer', 'wimbledon', 2017, -4, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('roger-federer', 'wimbledon', 2017, -3, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('roger-federer', 'wimbledon', 2017, -2, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('roger-federer', 'wimbledon', 2017, -1, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('roger-federer', 'wimbledon', 2017, -0, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('roger-federer', 'wimbledon', 2017, 1, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('roger-federer', 'wimbledon', 2017, 2, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('roger-federer', 'wimbledon', 2017, 3, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('roger-federer', 'wimbledon', 2017, 4, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('roger-federer', 'wimbledon', 2017, 5, True, abs_probabilities_per_surface, 'Clay', True))


print(spread_prob('rafael-nadal', 'wimbledon', 2017, -4, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('rafael-nadal', 'wimbledon', 2017, -3, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('rafael-nadal', 'wimbledon', 2017, -2, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('rafael-nadal', 'wimbledon', 2017, -1, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('rafael-nadal', 'wimbledon', 2017, -0, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('rafael-nadal', 'wimbledon', 2017, 1, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('rafael-nadal', 'wimbledon', 2017, 2, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('rafael-nadal', 'wimbledon', 2017, 3, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('rafael-nadal', 'wimbledon', 2017, 4, True, abs_probabilities_per_surface, 'Clay', True))
print(spread_prob('rafael-nadal', 'wimbledon', 2017, 5, True, abs_probabilities_per_surface, 'Clay', True))


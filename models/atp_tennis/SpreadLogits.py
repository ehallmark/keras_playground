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


def spread_prob(player, tournament, year, spread, is_grand_slam, surface='Hard', win=True, alpha=10.0):
    if is_grand_slam:
        r = range(-18, 19, 1)
        if win:
            prior = abs_probabilities_per_surface[surface][2]
            sql = slam_wins
        else:
            prior = abs_probabilities_per_surface[surface][3]
            sql = slam_losses
    else:
        r = range(-12, 13, 1)
        if win:
            prior = abs_probabilities_per_surface[surface][0]
            sql = wins
        else:
            prior = abs_probabilities_per_surface[surface][1]
            sql = losses
    row = sql[((sql.player_id == player) & (sql.tournament == tournament) & (sql.year == year))]
    probabilities = prior.copy()
    if row.shape[0] > 0:
        for k in probabilities:
            probabilities[k] *= alpha * 100.0
        print('prior: ', probabilities)
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

    print('posterior: ', probabilities)
    probabilities_over = {}
    for k in probabilities:
        probabilities_over[k] = 0.
        for j in probabilities_over:
            if j > k:
                probabilities_over[k] += probabilities[j]

    return probabilities_over[-int(spread)]


print(spread_prob('roger-federer', 'wimbledon', 2018, 2, True, 'Clay', True))


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sqlalchemy import create_engine
import statsmodels.formula.api as smf
from models.atp_tennis.SpreadMonteCarlo import abs_probabilities_per_surface, abs_game_total_probabilities_per_surface, abs_set_total_probabilities_per_surface
np.random.seed(1)


conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
slam_wins = pd.read_sql('select * from atp_matches_spread_probabilities_slam_wins', conn)
slam_losses = pd.read_sql('select * from atp_matches_spread_probabilities_slam_losses', conn)
wins = pd.read_sql('select * from atp_matches_spread_probabilities_win', conn)
losses = pd.read_sql('select * from atp_matches_spread_probabilities_losses', conn)
slam_wins.set_index(['player_id', 'tournament', 'year'], inplace=True)
slam_losses.set_index(['player_id', 'tournament', 'year'], inplace=True)
wins.set_index(['player_id', 'tournament', 'year'], inplace=True)
losses.set_index(['player_id', 'tournament', 'year'], inplace=True)

game_totals_slam = pd.read_sql('select * from atp_matches_total_probabilities_slam', conn)
game_totals = pd.read_sql('select * from atp_matches_total_probabilities', conn)
game_totals_slam.set_index(['player_id', 'tournament', 'year'], inplace=True)
game_totals.set_index(['player_id', 'tournament', 'year'], inplace=True)

set_totals_slam_win = pd.read_sql('select * from atp_matches_set_total_probabilities_slam_win', conn)
set_totals_win = pd.read_sql('select * from atp_matches_set_total_probabilities_win', conn)
set_totals_slam_win.set_index(['player_id', 'tournament', 'year'], inplace=True)
set_totals_win.set_index(['player_id', 'tournament', 'year'], inplace=True)

set_totals_slam_loss = pd.read_sql('select * from atp_matches_set_total_probabilities_slam_loss', conn)
set_totals_loss = pd.read_sql('select * from atp_matches_set_total_probabilities_loss', conn)
set_totals_slam_loss.set_index(['player_id', 'tournament', 'year'], inplace=True)
set_totals_loss.set_index(['player_id', 'tournament', 'year'], inplace=True)



def build_cumulative_probabilities(probabilities):
    probabilities_over = {}
    for k in probabilities:
        probabilities_over[k] = 0.
        for j in probabilities:
            if j > k:
                probabilities_over[k] += probabilities[j]
    return probabilities_over


def spread_prob(player, tournament, year, spread, is_grand_slam, priors_per_surface, surface='Hard', win=True, alpha=5.0):
    if math.isnan(spread):
        return np.NaN
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

    try:
        row = sql.loc[(player, tournament, int(year)), :]
    except KeyError as e:
        row = np.array([])

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


def total_games_prob(player, tournament, year, total, is_grand_slam, priors_per_surface, surface='Hard', under=True, win=True, alpha=5.0):
    if math.isnan(total):
        return np.NaN
    if is_grand_slam:
        r = range(1, 67, 1)
        if win:
            prior = priors_per_surface[2]
        else:
            prior = priors_per_surface[3]
        sql = game_totals_slam
    else:
        r = range(1, 41, 1)
        if win:
            prior = priors_per_surface[0]
        else:
            prior = priors_per_surface[1]
        sql = game_totals

    try:
        row = sql.loc[(player, tournament, int(year)), :]
    except KeyError as e:
        row = np.array([])

    probabilities = prior.copy()
    if row.shape[0] > 0:
        for k in probabilities:
            probabilities[k] *= alpha * 100.0
        for i in r:
            x = int(row['plus' + str(i)])
            probabilities[i] += x

        s = 0.0
        for _, v in probabilities.items():
            s += v

        if s > 0:
            for k in probabilities:
                probabilities[k] /= s

    probabilities_over = build_cumulative_probabilities(probabilities)
    if is_grand_slam:
        if total > 66:
            total = 66
    else:
        if total > 40:
            total = 40
    if under:
        prob = 1.0 - probabilities_over[int(total-0.5)]
    else:
        prob = probabilities_over[int(total)]
    return prob


def total_sets_prob(player, tournament, year, total, is_grand_slam, priors_per_surface, surface='Hard', win=True, under=True, alpha=5.0):
    if math.isnan(total):
        return np.NaN
    if is_grand_slam:
        r = range(3, 6, 1)
        if win:
            prior = priors_per_surface[2]
            sql = set_totals_slam_win
        else:
            prior = priors_per_surface[3]
            sql = set_totals_slam_loss
    else:
        r = range(2, 4, 1)
        if win:
            prior = priors_per_surface[0]
            sql = set_totals_win
        else:
            prior = priors_per_surface[1]
            sql = set_totals_loss
    try:
        row = sql.loc[(player, tournament, int(year)), :]
    except KeyError as e:
        row = np.array([])

    probabilities = prior.copy()
    if row.shape[0] > 0:
        for k in probabilities:
            probabilities[k] *= alpha * 100.0
        for i in r:
            x = int(row['plus' + str(i)])
            probabilities[i] += x

        s = 0.0
        for _, v in probabilities.items():
            s += v

        if s > 0:
            for k in probabilities:
                probabilities[k] /= s

    probabilities_over = build_cumulative_probabilities(probabilities)
    if is_grand_slam:
        if total > 5:
            return np.NaN
        if total < 3:
            return np.NaN
    else:
        if total > 3:
            return np.NaN
        if total < 2:
            return np.NaN

    if under:
        if total < 2.5:
            return 0
        prob = 1.0 - probabilities_over[int(total-0.5)]
    else:
        prob = probabilities_over[int(total)]
    return prob


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


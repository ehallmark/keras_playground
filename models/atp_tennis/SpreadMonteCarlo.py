import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

np.random.seed(1)


def simulate_match(best_of):
    def simulate_set(odds1, odds2):
        s1, s2 = 0, 0
        for i in range(13):
            if s1 == 7 or s2 == 7 or (s1 == 6 and s2 <= 4) or (s2 == 6 and s1 <= 4):
                break
            if i % 2 == 0:
                odds = odds1
            else:
                odds = odds2
            if np.random.rand(1) < odds:
                s1 += 1
            else:
                s2 += 1
        return s1, s2

    m1, m2 = 0, 0
    spread = 0
    best_to = int(best_of/2)+1
    odds1 = 0.5
    odds2 = 0.5
    for i in range(best_of):
        if m1 >= best_to or m2 >= best_to:
            break
        if abs(spread) % 2 == 0:
            _odds1, _odds2 = odds1, odds2
        else:
            _odds1, _odds2 = odds2, odds1
        s1, s2 = simulate_set(_odds1, _odds2)
        spread += s1 - s2
        if s1 > s2:
            m1 += 1
        else:
            m2 += 1
    return m1, m2, spread


def create_probabilities(query, use_monte_carlo=False):
    x = []
    x_3 = []
    x_loss = []
    x_3_loss = []
    if use_monte_carlo:
        for i in range(50000):
            _, _, spread = simulate_match(5)
            x.append(spread)
            x.append(-spread)
            _, _, spread = simulate_match(3)
            x_3.append(spread)
            x_3.append(-spread)
    else:
        conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
        sql = pd.read_sql(query, conn)
        for i in range(sql.shape[0]):
            row = sql.iloc[i]
            victory = row['player_victory']
            spread = int(row['spread'])
            if victory:
                if row['grand_slam'] > 0.5:
                    x.append(spread)
                else:
                    x_3.append(spread)
            else:
                if row['grand_slam'] > 0.5:
                    x_loss.append(spread)
                else:
                    x_3_loss.append(spread)

    probabilities5 = {}
    probabilities3 = {}

    for i in range(-18, 19):
        probabilities5[i] = 0.0

    for i in range(-12, 13):
        probabilities3[i] = 0.0

    for p in x:
        probabilities5[int(p)] += 1
    for p in x_3:
        probabilities3[int(p)] += 1

    for k in probabilities3:
        probabilities3[k] /= len(x_3)

    for k in probabilities5:
        probabilities5[k] /= len(x)

    probabilities5_loss = {}
    probabilities3_loss = {}

    for i in range(-18, 19):
        probabilities5_loss[i] = 0.0

    for i in range(-12, 13):
        probabilities3_loss[i] = 0.0

    for p in x_loss:
        probabilities5_loss[int(p)] += 1
    for p in x_3_loss:
        probabilities3_loss[int(p)] += 1

    for k in probabilities3_loss:
        probabilities3_loss[k] /= len(x_3_loss)

    for k in probabilities5_loss:
        probabilities5_loss[k] /= len(x_loss)

    return probabilities3, probabilities3_loss, probabilities5, probabilities5_loss


def create_spread_for_query(query, use_monte_carlo=False):
    probabilities3, probabilities3_loss, probabilities5, probabilities5_loss = create_probabilities(query, use_monte_carlo)
    probabilities5_under = {}
    probabilities3_under = {}
    probabilities5_over = {}
    probabilities3_over = {}

    probabilities5_under_loss = {}
    probabilities3_under_loss = {}
    probabilities5_over_loss = {}
    probabilities3_over_loss = {}

    for k in probabilities3:
        probabilities3_over[k] = 0.
        probabilities3_under[k] = 0.
        for j in probabilities3:
            if j < k:
                probabilities3_under[k] += probabilities3[j]
            if j > k:
                probabilities3_over[k] += probabilities3[j]

    for k in probabilities5:
        probabilities5_over[k] = 0.
        probabilities5_under[k] = 0.
        for j in probabilities5:
            if j < k:
                probabilities5_under[k] += probabilities5[j]
            if j > k:
                probabilities5_over[k] += probabilities5[j]

    for k in probabilities3_loss:
        probabilities3_over_loss[k] = 0.
        probabilities3_under_loss[k] = 0.
        for j in probabilities3_loss:
            if j < k:
                probabilities3_under_loss[k] += probabilities3_loss[j]
            if j > k:
                probabilities3_over_loss[k] += probabilities3_loss[j]

    for k in probabilities5_loss:
        probabilities5_over_loss[k] = 0.
        probabilities5_under_loss[k] = 0.
        for j in probabilities5_loss:
            if j < k:
                probabilities5_under_loss[k] += probabilities5_loss[j]
            if j > k:
                probabilities5_over_loss[k] += probabilities5_loss[j]
    return probabilities3_over, probabilities5_over, probabilities3_over_loss, probabilities5_over_loss


sql_str = '''
        select player_victory,games_won-games_against as spread, case when greatest(num_sets-sets_won,sets_won)=3 or tournament in ('roland-garros','wimbledon','us-open','australian-open')
                then 1.0 else 0.0 end as grand_slam from atp_matches_individual where year >= 1991 and year <= 2010 and tournament is not null and num_sets is not null and sets_won is not null and games_won is not null
    '''

sql_clay = sql_str + ' and court_surface=\'Clay\''
sql_grass = sql_str + ' and court_surface=\'Grass\''
sql_hard = sql_str + ' and court_surface=\'Hard\''


probabilities_per_surface = {
    'Clay': create_spread_for_query(sql_clay),
    'Grass': create_spread_for_query(sql_grass),
    'Hard': create_spread_for_query(sql_hard)
}


abs_probabilities_per_surface = {
    'Clay': create_probabilities(sql_clay),
    'Grass': create_probabilities(sql_grass),
    'Hard': create_probabilities(sql_hard)
}


def probability_beat_given_win(spread, court_surface, grand_slam=False):
    if grand_slam:
        return probabilities_per_surface[court_surface][1][-int(spread)]
    else:
        return probabilities_per_surface[court_surface][0][-int(spread)]


def probability_beat_given_loss(spread, court_surface, grand_slam=False):
    if grand_slam:
        return probabilities_per_surface[court_surface][3][-int(spread)]
    else:
        return probabilities_per_surface[court_surface][2][-int(spread)]


if __name__ == '__main__':
    for surf in ['Clay', 'Hard', 'Grass']:
        print('Given WIN', surf)
        print('Beat 5', probability_beat_given_win(5, surf))
        print('Beat 4.5', probability_beat_given_win(4.5, surf))
        print('Beat 4', probability_beat_given_win(4, surf))
        print('Beat 3', probability_beat_given_win(3, surf))
        print('Beat 2', probability_beat_given_win(2, surf))
        print('Beat 1', probability_beat_given_win(1, surf))
        print('Beat 0', probability_beat_given_win(0, surf))
        print('Beat -1', probability_beat_given_win(-1, surf))
        print('Beat -2', probability_beat_given_win(-2, surf))
        print('Beat -3', probability_beat_given_win(-3, surf))
        print('Beat -4', probability_beat_given_win(-4, surf))
        print('Beat -4.5', probability_beat_given_win(-4.5, surf))
        print('Beat -5', probability_beat_given_win(-5, surf))
        print('Beat -6', probability_beat_given_win(-6, surf))

        print('Given LOSS')
        print('Beat 5', probability_beat_given_loss(5, surf))
        print('Beat 4.5', probability_beat_given_loss(4.5, surf))
        print('Beat 4', probability_beat_given_loss(4, surf))
        print('Beat 3', probability_beat_given_loss(3, surf))
        print('Beat 2', probability_beat_given_loss(2, surf))
        print('Beat 1', probability_beat_given_loss(1, surf))
        print('Beat 0', probability_beat_given_loss(0, surf))
        print('Beat -1', probability_beat_given_loss(-1, surf))
        print('Beat -2', probability_beat_given_loss(-2, surf))
        print('Beat -3', probability_beat_given_loss(-3, surf))
        print('Beat -4', probability_beat_given_loss(-4, surf))
        print('Beat -4.5', probability_beat_given_loss(-4.5, surf))
        print('Beat -5', probability_beat_given_loss(-5, surf))
        print('Beat -6', probability_beat_given_loss(-6, surf))


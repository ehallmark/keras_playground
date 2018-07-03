import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

np.random.seed(1)
conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")


def create_probabilities(query, _min, _max, y, is_grand_slam=False):
    x = []
    x_loss = []
    sql = pd.read_sql(query, conn)
    if is_grand_slam:
        sql = sql[sql.grand_slam > 0.5]
    else:
        sql = sql[sql.grand_slam < 0.5]

    for i in range(sql.shape[0]):
        row = sql.iloc[i]
        victory = row['player_victory']
        spread = int(row[y])
        if victory:
            x.append(spread)
        else:
            x_loss.append(spread)

    probabilities = {}
    for i in range(_min, _max+1):
        probabilities[i] = 0.0

    for p in x:
        probabilities[min(_max, int(p))] += 1

    for k in probabilities:
        probabilities[k] /= len(x)

    probabilities_loss = {}

    for i in range(_min, _max+1):
        probabilities_loss[i] = 0.0

    for p in x_loss:
        probabilities_loss[min(_max, int(p))] += 1

    for k in probabilities_loss:
        probabilities_loss[k] /= len(x_loss)

    return probabilities, probabilities_loss


def build_cumulative_probabilities(probabilities):
    probabilities_over = {}
    for k in probabilities:
        probabilities_over[k] = 0.
        for j in probabilities:
            if j > k:
                probabilities_over[k] += probabilities[j]
    return probabilities_over


def create_spread_for_query(query):
    probabilities3, probabilities3_loss, probabilities5, probabilities5_loss \
        = create_spread_probabilities_from_query(query)
    probabilities5_over = build_cumulative_probabilities(probabilities5)
    probabilities3_over = build_cumulative_probabilities(probabilities3)
    probabilities5_over_loss = build_cumulative_probabilities(probabilities5_loss)
    probabilities3_over_loss = build_cumulative_probabilities(probabilities3_loss)
    return probabilities3_over, probabilities5_over, probabilities3_over_loss, probabilities5_over_loss


def create_totals_for_query(query):
    probabilities3, probabilities5 \
        = create_totals_probabilities_from_query(query)
    probabilities5_over = build_cumulative_probabilities(probabilities5)
    probabilities3_over = build_cumulative_probabilities(probabilities3)
    return probabilities3_over, probabilities5_over


def create_spread_probabilities_from_query(query):
    probabilities3, probabilities3_loss = create_probabilities(query, -12, 12, 'spread', False)
    probabilities5, probabilities5_loss = create_probabilities(query, -18, 18, 'spread', True)
    return probabilities3, probabilities3_loss, probabilities5, probabilities5_loss


def create_totals_probabilities_from_query(query):
    probabilities3, _ = create_probabilities(query, 0, 39, 'total', False)
    probabilities5, _ = create_probabilities(query, 0, 70, 'total', True)
    return probabilities3, probabilities5


sql_str = '''
        select player_victory,games_won-games_against as spread, case when greatest(num_sets-sets_won,sets_won)=3 or tournament in ('roland-garros','wimbledon','us-open','australian-open')
                then 1.0 else 0.0 end as grand_slam from atp_matches_individual where year >= 1991 and year <= 2010 and tournament is not null and num_sets is not null and sets_won is not null and games_won is not null
    '''

sql_clay = sql_str + ' and court_surface=\'Clay\''
sql_grass = sql_str + ' and court_surface=\'Grass\''
sql_hard = sql_str + ' and court_surface=\'Hard\''


sql_total_str = '''
        select player_victory,games_won+games_against as total, case when greatest(num_sets-sets_won,sets_won)=3 or tournament in ('roland-garros','wimbledon','us-open','australian-open')
                then 1.0 else 0.0 end as grand_slam from atp_matches_individual where year >= 1991 and year <= 2010 and tournament is not null and num_sets is not null and sets_won is not null and games_won is not null
    '''

sql_total_clay = sql_total_str + ' and court_surface=\'Clay\''
sql_total_grass = sql_total_str + ' and court_surface=\'Grass\''
sql_total_hard = sql_total_str + ' and court_surface=\'Hard\''


probabilities_per_surface = {
    'Clay': create_spread_for_query(sql_clay),
    'Grass': create_spread_for_query(sql_grass),
    'Hard': create_spread_for_query(sql_hard)
}


abs_probabilities_per_surface = {
    'Clay': create_spread_probabilities_from_query(sql_clay),
    'Grass': create_spread_probabilities_from_query(sql_grass),
    'Hard': create_spread_probabilities_from_query(sql_hard)
}


total_probabilities_per_surface = {
    'Clay': create_totals_for_query(sql_total_clay),
    'Grass': create_totals_for_query(sql_total_grass),
    'Hard': create_totals_for_query(sql_total_hard)
}

abs_total_probabilities_per_surface = {
    'Clay': create_totals_probabilities_from_query(sql_total_clay),
    'Grass': create_totals_probabilities_from_query(sql_total_grass),
    'Hard': create_totals_probabilities_from_query(sql_total_hard)
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


def probability_beat_total(total, court_surface, grand_slam=False):
    if grand_slam:
        return total_probabilities_per_surface[court_surface][1][int(total)]
    else:
        return total_probabilities_per_surface[court_surface][0][int(total)]


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

    for surf in ['Clay', 'Hard', 'Grass']:
        print('TOTALS: Given WIN', surf)
        print('Beat 25', probability_beat_total(25, surf))
        print('Beat 20', probability_beat_total(20, surf))
        print('Beat 15', probability_beat_total(15, surf))
        print('Beat 13', probability_beat_total(13, surf))
        print('Beat 12', probability_beat_total(12, surf))
        print('Beat 11', probability_beat_total(11, surf))
        print('Beat 10', probability_beat_total(10, surf))

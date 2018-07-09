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


'''
def create_spread_for_query(query):
    probabilities3, probabilities3_loss, probabilities5, probabilities5_loss \
        = create_spread_probabilities_from_query(query)
    probabilities5_over = build_cumulative_probabilities(probabilities5)
    probabilities3_over = build_cumulative_probabilities(probabilities3)
    probabilities5_over_loss = build_cumulative_probabilities(probabilities5_loss)
    probabilities3_over_loss = build_cumulative_probabilities(probabilities3_loss)
    return probabilities3_over, probabilities5_over, probabilities3_over_loss, probabilities5_over_loss
'''


def create_spread_probabilities_from_query(query):
    probabilities3, probabilities3_loss = create_probabilities(query, -12, 12, 'spread', False)
    probabilities5, probabilities5_loss = create_probabilities(query, -18, 18, 'spread', True)
    return probabilities3, probabilities3_loss, probabilities5, probabilities5_loss


def create_game_totals_probabilities_from_query(query):
    probabilities3, probabilities3_loss = create_probabilities(query, 0, 40, 'total', False)
    probabilities5, probabilities5_loss = create_probabilities(query, 0, 66, 'total', True)
    return probabilities3, probabilities3_loss, probabilities5, probabilities5_loss


def create_set_totals_probabilities_from_query(query):
    probabilities3, probabilities3_loss = create_probabilities(query, 2, 3, 'total', False)
    probabilities5, probabilities5_loss = create_probabilities(query, 3, 5, 'total', True)
    return probabilities3, probabilities3_loss, probabilities5, probabilities5_loss


sql_str = '''
        select player_victory,games_won-games_against as spread, case when greatest(num_sets-sets_won,sets_won)=3 or tournament in ('roland-garros','wimbledon','us-open','australian-open')
                then 1.0 else 0.0 end as grand_slam from atp_matches_individual where year >= 1991 and year <= 2010 and tournament is not null and num_sets is not null and sets_won is not null and games_won is not null
    '''

sql_clay = sql_str + ' and court_surface=\'Clay\''
sql_grass = sql_str + ' and court_surface=\'Grass\''
sql_hard = sql_str + ' and court_surface=\'Hard\''

sql_game_total_str = '''
        select player_victory,games_won+games_against as total, case when greatest(num_sets-sets_won,sets_won)=3 or tournament in ('roland-garros','wimbledon','us-open','australian-open')
                then 1.0 else 0.0 end as grand_slam from atp_matches_individual where year >= 1991 and year <= 2010 and tournament is not null and num_sets is not null and sets_won is not null and games_won is not null
    '''

sql_set_total_str = '''
        select player_victory,num_sets as total, case when greatest(num_sets-sets_won,sets_won)=3 or tournament in ('roland-garros','wimbledon','us-open','australian-open')
                then 1.0 else 0.0 end as grand_slam from atp_matches_individual where year >= 1991 and year <= 2010 and tournament is not null and num_sets is not null and sets_won is not null and games_won is not null
    '''

abs_probabilities_per_surface = {
    'Clay': create_spread_probabilities_from_query(sql_clay),
    'Grass': create_spread_probabilities_from_query(sql_grass),
    'Hard': create_spread_probabilities_from_query(sql_hard)
}

abs_game_total_probabilities_per_surface = \
    create_game_totals_probabilities_from_query(sql_game_total_str)

abs_set_total_probabilities_per_surface = \
    create_set_totals_probabilities_from_query(sql_set_total_str)

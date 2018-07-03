import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import statsmodels.formula.api as smf


np.random.seed(1)
sql_str = '''
        select case when player_victory then 1.0 else 0.0 end as player_victory,
        games_won-games_against as spread, 
        case when court_surface='Clay' then 1.0 else 0.0 end as clay,
        case when court_surface='Grass' then 1.0 else 0.0 end as grass, 
        case when greatest(num_sets-sets_won,sets_won)=3 or tournament in ('roland-garros','wimbledon','us-open','australian-open')
                then 1.0 else 0.0 end as grand_slam 
        from atp_matches_individual where year >= 1991 and year <= 2010 and tournament is not null and num_sets is not null and sets_won is not null and games_won is not null
    '''

conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
sql = pd.read_sql(sql_str, conn)

grand_slam_data = sql[sql.grand_slam > 0.5]
regular_data = sql[sql.grand_slam < 0.5]

grand_slam_wins = grand_slam_data[grand_slam_data.player_victory > 0.5]
grand_slam_losses = grand_slam_data[grand_slam_data.player_victory < 0.5]

regular_wins = regular_data[regular_data.player_victory > 0.5]
regular_losses = regular_data[regular_data.player_victory < 0.5]


def spread_func(x, i):
    if x > i:
        return 1.0
    else:
        return 0.0


def train(data, min, max):
    for i in range(min, max+1, 1):
        print('Calculating odds for spread: ', i)
        data['y'] = [spread_func(x, i) for x in data['spread']]
        try:
            results = smf.logit('y ~ ' + '+'.join(['clay', 'grass']), data=data).fit()
            print(results.summary())
        except:
            print('Unable to predict: ', i)


train(grand_slam_wins, -10, 10)
train(grand_slam_losses, -10, 10)
train(regular_wins, -6, 6)
train(regular_losses, -6, 6)


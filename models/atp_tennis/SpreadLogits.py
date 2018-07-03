import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import statsmodels.formula.api as smf


np.random.seed(1)
sql_str = '''
        select * from atp_matches_spread_probabilities_win '''

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


import pandas as pd
from sqlalchemy import create_engine
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
'''

'''


def test_model(model, x, y):
    predictions = model.predict(x)
    binary_predictions = (predictions >= 0.5).astype(int)
    binary_errors = np.array(y) != np.array(binary_predictions)
    binary_errors = binary_errors.astype(int)
    errors = np.array(y - np.array(predictions))
    binary_correct = predictions.shape[0] - int(binary_errors.sum())
    binary_percent = float(binary_correct) / predictions.shape[0]
    avg_error = np.mean(np.abs(errors), -1)
    return binary_correct, y.shape[0], binary_percent, avg_error


def to_percentage(x):
    return str("%0.2f" % (x * 100))+'%'


def bool_to_int(b):
    if b:
        return 1.0
    else:
        return 0.0


conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
sql = pd.read_sql('select p1.*,p2.plus_minus::double precision/greatest(1,p2.num_games_played) as plus_minus0, '+
                  'p2.fg_pct as fg_pct0, '+
                  'p2.tov as tov0, p2.pts as pts0, p2.oreb as oreb0, p2.fga as fga0, p2.fgm as fgm0 '
                  'from nba_players_game_stats as p1 join nba_players_game_stats_month_lag as p2 '
                  'on ((p1.player_id,p1.game_id)=(p2.player_id,p2.game_id)) where p1.game_date is not null and ' +
                  'p1.season_type = \'Regular Season\' and p1.fg3_pct is ' +
                  'not null and p1.stl is not null and p1.oreb is not null and p1.plus_minus is not null '+
                  'order by p1.game_date asc', conn)

test_season = 2017

#print('Data: ', sql[:100])

wins = [bool_to_int(y == 'W') for y in sql['wl']]
print("Num wins: "+str(len(wins)))
print("Num datapoints: ",str(sql.shape[0]))
sql['y'] = wins
sql['home'] = [bool_to_int(not '@' in matchup) for matchup in sql['matchup']]

# fill in nans and infs
#sql[sql.stl == np.nan] = 1.0
#sql[sql.stl == np.inf] = 1.0
#sql[sql.fg3m == np.nan] = 1.0
#sql[sql.fg3m == np.inf] = 1.0

input_attributes = [
    'home',
    'oreb',
    'dreb',
    'tov',
    'blk',
    'fga',
    'fgm',
    'ast',
    #'fg3_pct',
    'fg3a',
    'fg3m',
    'pf',
    'min',
    'fta',
    'plus_minus',
    'plus_minus0',
    'fgm0',
    'fga0',
    'tov0',
    'oreb0',
    'pts0'
]
all_attributes = list(input_attributes)
all_attributes.append('y')
all_attributes.append('season_year')

sql = sql[all_attributes].astype(np.float64)

test_data = sql[sql.season_year == test_season]
sql = sql[sql.season_year != test_season]

print('Attrs: ', sql[input_attributes])

# model to predict the total score (h_pts + a_pts)
results = smf.logit('y ~ '+'+'.join(input_attributes), data=sql).fit()
print(results.summary())

binary_correct, n, binary_percent, avg_error = test_model(results, test_data, test_data['y'])

print('Correctly predicted: '+str(binary_correct)+' out of '+str(n) +
      ' ('+to_percentage(binary_percent)+')')
print('Average error: ', to_percentage(avg_error))

exit(0)


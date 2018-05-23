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
sql = pd.read_sql('select * from nba_games_all where game_date is not null and ' +
                  'season_type = \'Regular Season\' and h_fg3_pct is ' +
                  'not null and a_fg3_pct is not null and a_stl is not null and h_oreb is not null '+
                  'order by game_date asc', conn)

test_season = 2017

sql['spread'] = sql['h_pts'].astype(np.float64) - sql['a_pts'].astype(np.float64)
sql['total'] = sql['h_pts'].astype(np.float64) + sql['a_pts'].astype(np.float64)
sql['y'] = [bool_to_int(y >= 0) for y in sql['spread']]
sql['stl'] = sql['h_stl'].astype(np.float64) - sql['a_stl'].astype(np.float64)
sql['oreb'] = sql['h_oreb'].astype(np.float64) - sql['a_oreb'].astype(np.float64)
sql['tov'] = sql['h_tov'].astype(np.float64) - sql['a_tov'].astype(np.float64)
sql['fg_pct'] = sql['h_fg_pct'].astype(np.float64) - sql['a_fg_pct'].astype(np.float64)
sql['fg3m'] = sql['h_fg3m'].astype(np.float64) - sql['a_fg3m'].astype(np.float64)
sql['fg3_pct'] = sql['h_fg3_pct'].astype(np.float64) - sql['a_fg3_pct'].astype(np.float64)
sql['fta'] = sql['h_fta'].astype(np.float64) - sql['a_fta'].astype(np.float64)
sql['pf'] = sql['h_pf'].astype(np.float64) - sql['a_pf'].astype(np.float64)
sql['ast'] = sql['h_ast'].astype(np.float64) - sql['a_ast'].astype(np.float64)
sql['fg3a'] = sql['h_fg3a'].astype(np.float64) - sql['a_fg3a'].astype(np.float64)
sql['fga'] = sql['h_fga'].astype(np.float64) - sql['a_fga'].astype(np.float64)
sql['fgm'] = sql['h_fgm'].astype(np.float64) - sql['a_fgm'].astype(np.float64)
sql['ft_pct'] = sql['h_ft_pct'].astype(np.float64) - sql['a_ft_pct'].astype(np.float64)

# fill in nans and infs
sql[sql.stl == np.nan] = 1.0
sql[sql.stl == np.inf] = 1.0
sql[sql.fg3m == np.nan] = 1.0
sql[sql.fg3m == np.inf] = 1.0

input_attributes = [
    # 'h_stl', 'h_oreb', 'h_tov', 'h_fgm', 'h_fga', 'h_fg3a', 'h_fg3m', 'h_pf',
    # 'a_stl', 'a_oreb', 'a_tov', 'a_fgm', 'a_fga', 'a_fg3a', 'a_fg3m', 'a_fta'
    'stl', 'oreb', 'tov', 'fgm', 'fga', 'ast', 'fg3a', 'fg3m', 'a_pf'  # How big of an issue is quasi-separation???
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


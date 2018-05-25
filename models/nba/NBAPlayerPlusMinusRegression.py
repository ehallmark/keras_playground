import pandas as pd
from sqlalchemy import create_engine
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
'''

'''


def test_model(model, x, y):
    predictions = model.predict(x)
    errors = np.array(y - np.array(predictions))
    avg_error = np.mean(np.abs(errors), -1)
    return errors, avg_error, y.shape[0]


def to_percentage(x):
    return str("%0.2f" % (x * 100))+'%'


def bool_to_int(b):
    if b:
        return 1.0
    else:
        return 0.0


conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")

query = '''
select p1.*,
  (p2.plus_minus)::double precision/greatest(1,p2.num_games_played) as plus_minus0, 
  p2.num_games_played as n, p2.tov as tov0, p2.ast as ast0, p2.pts as pts0, p2.oreb as oreb0, p2.fga as fga0, p2.fgm as fgm0, 
  p2.min::double precision/p2.num_games_played as min0,
  (p3.plus_minus)::double precision/greatest(1,p3.num_games_played) as plus_minus1, 
  p3.num_games_played as n1, p3.tov as tov1, p3.ast as ast1, p3.pts as pts1, p3.oreb as oreb1, p3.fga as fga1, p3.fgm as fgm1, 
  p3.min::double precision/p3.num_games_played as min1 
  from nba_players_game_stats as p1 
  join nba_players_game_stats_month_lag as p2 
  on ((p1.player_id,p1.game_id)=(p2.player_id,p2.game_id)) 
  join nba_players_game_stats_previous_matchups as p3
  on ((p1.player_id,p1.game_id)=(p3.player_id,p3.game_id)) where p1.game_date is not null and 
  p1.season_type = \'Regular Season\' and p1.fg3_pct is 
  not null and p1.stl is not null and p1.oreb is not null and p1.plus_minus is not null 
  order by p1.game_date asc
'''
sql = pd.read_sql(query, conn)

test_season = 2017

sql['home'] = [bool_to_int(not '@' in matchup) for matchup in sql['matchup']]

# fill in nans and infs
#sql[sql.stl == np.nan] = 1.0
#sql[sql.stl == np.inf] = 1.0
#sql[sql.fg3m == np.nan] = 1.0
#sql[sql.fg3m == np.inf] = 1.0

input_attributes = [
    'home',
    #'oreb',
    #'dreb',
    #'tov',
    #'blk',
    #'fga',
    #'fgm',
    #'ast',
    #'fg3_pct',
    #'fg3a',
    #'fg3m',
    #'pf',
    #'min',
    #'fta',
    'plus_minus0',
    'plus_minus1',
    #'tov0',
    #'tov1',
    #'oreb0',
    #'oreb1',
    'fgm0',
    #'fgm1',
    'fga0',
    #'fga1',
    #'ast0',
    'ast1',
    #'n',
    #'n1',
    #'min0',
    #'min1'
]
all_attributes = list(input_attributes)
all_attributes.append('plus_minus')
all_attributes.append('season_year')

sql = sql[all_attributes].astype(np.float64)

test_data = sql[sql.season_year == test_season]
sql = sql[sql.season_year != test_season]

print('Attrs: ', sql[input_attributes])

# model to predict the total score (h_pts + a_pts)
results = smf.ols('plus_minus ~ '+'+'.join(input_attributes), data=sql).fit()
print(results.summary())

errors, avg_error, n = test_model(results, test_data, test_data['plus_minus'])
print('Average error (n_test='+str(n)+'): ', avg_error)

plt.figure()
lines_true = plt.plot(errors, color='b')
plt.show()

exit(0)


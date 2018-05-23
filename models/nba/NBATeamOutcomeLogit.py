import pandas as pd
from sqlalchemy import create_engine
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
'''

'''

def bool_to_int(b):
    if b:
        return 1.0
    else:
        return 0.0

#team_id = '1610612739'  # cleveland  '1610612757' # portland
conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
sql = pd.read_sql('select * from nba_games_all where game_date is not null and ' +
                  'season_type = \'Regular Season\' and h_fg3_pct is ' +
                  'not null and a_fg3_pct is not null '+
                 # 'and \''+ team_id+'\'= ANY(ARRAY[h_team_id]) ' +
                 # 'and season_year >= 2010 ' +
                  'order by game_date asc', conn)

test_season = 2017

sql['spread'] = sql['h_pts'] - sql['a_pts']
sql['total'] = sql['h_pts'] + sql['a_pts']
sql['y'] = [bool_to_int(y >= 0) for y in sql['spread']]
sql['stl'] = sql['h_stl'] - sql['a_stl']
sql['oreb'] = sql['h_oreb'] - sql['a_oreb']
sql['tov'] = sql['h_tov'] - sql['a_tov']
sql['fg_pct'] = sql['h_fg_pct'] - sql['a_fg_pct']
sql['fg3m'] = sql['h_fg3m'] - sql['a_fg3m']
sql['fta'] = sql['h_fta'] - sql['a_fta']
sql['pf'] = sql['h_pf'] - sql['a_pf']
sql['total_pf'] = sql['h_pf'] + sql['a_pf']

#total = []
#spread = []
#outcome = []
#for i in range(len(sql)):
#    if i > 0:
#        total.append(sql['total'][i])
#        spread.append(sql['spread'][i])
#        outcome.append(sql['y'][i])
#sql = sql[:-1]
#sql['spread'] = spread
#sql['total'] = total
#sql['y'] = outcome

test_data = sql[sql.season_year == test_season]
sql = sql[sql.season_year != test_season]
input_attributes = [
    #'h_stl', 'h_oreb', 'h_tov', 'h_fg_pct', 'h_fg3m', 'h_fta',
    #'a_stl', 'a_oreb', 'a_tov', 'a_fg_pct', 'a_fg3m', 'a_fta'
    'stl', 'oreb', 'tov', 'fg_pct', 'fg3m', 'a_pf'
]

# model to predict the total score (h_pts + a_pts)
results = smf.logit('y ~ '+'+'.join(input_attributes), data=sql).fit()
print(results.summary())

predictions = results.predict(test_data)
binary_predictions = (predictions >= 0.5).astype(int)
binary_errors = np.array(test_data['y']) != np.array(binary_predictions)
binary_errors = binary_errors.astype(int)
errors = np.array(test_data['y'] - np.array(predictions))
binary_correct = predictions.shape[0]-int(binary_errors.sum())
binary_percent = float(binary_correct)/predictions.shape[0]
print('Correctly predicted: '+str(binary_correct)+' out of '+str(predictions.shape[0]) +
      ' ('+str("%0.2f" % (binary_percent*100))+'%)')
print('Average error: ', np.mean(np.abs(errors), -1))

exit(0)


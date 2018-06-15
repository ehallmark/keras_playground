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


test_season = 2017
start_season = 1990
conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
sql = pd.read_sql('''select * from nba_games_all  as h
            join nba_games_all as a on (h.game_id,h.a_team_id)=(a.game_id,a.team_id)
            where a.game_date is not null and a.season_type = \'Regular Season\' and 
            a.fg3_pct is not null and a.stl is not null and a.oreb is not null and 
            a.season_year>={{START}} and a.season_year<={{END}}
            order by a.game_date asc
'''.replace("{{START}}", str(start_season)).replace("{{END}}", str(test_season)), conn)

input_attributes = [
     'h_stl', 'h_oreb', 'h_tov', 'h_fgm', 'h_fga', 'h_fg3a', 'h_fg3m', 'h_pf',
     'a_stl', 'a_oreb', 'a_tov', 'a_fgm', 'a_fga', 'a_fg3a', 'a_fg3m', 'a_fta'
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




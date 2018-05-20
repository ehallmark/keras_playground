import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import statsmodels.formula.api as smf

plt.interactive(False)
conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")

player = 'LeBron James'
player = pd.read_sql('select person_id from nba_players_all where display_first_last = \''+player+'\'', conn)
player_id = player['person_id'][0]
print(player+' ID: ', player_id)

# get dataset for lebron james
sql = pd.read_sql('select * from nba_players_game_stats where player_id = \''+player_id+'\' and game_date is not null and '+
                  'season_type = \'Regular Season\' and min > 0 and fg3_pct is not null '+
                  'and season_year > 2010 '+
                  'order by game_date asc', conn)
test_year = 2017

sql['fga_pm'] = sql['fga']/sql['min']
#sql['game_date'] = [i for i in range(len(sql['game_date']))]
#print(sql['game_date'][:10])
sql_test = sql[sql.season_year == test_year]
sql = sql[sql.season_year != test_year]

attributes = ['fg_pct', 'oreb', 'stl', 'pts']
dependent_variable = 'fg_pct'

y = sql[attributes]
y_test = np.array(sql_test[attributes])
n_dim_state = len(attributes)

kf = KalmanFilter(n_dim_state=n_dim_state, n_dim_obs=len(attributes))
kf = kf.em(y, n_iter=10)
(initial_filtered_state_means, initial_filtered_state_covariances) = kf.filter(y_test)
n_timesteps = y_test.shape[0]
filtered_state_means = np.zeros((n_timesteps, n_dim_state))
filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
for t in range(n_timesteps - 1):
    if t == 0:
        filtered_state_means[t] = initial_filtered_state_means[len(initial_filtered_state_means)-1]
        filtered_state_covariances[t] = initial_filtered_state_covariances[len(initial_filtered_state_covariances)-1]
    filtered_state_means[t + 1], filtered_state_covariances[t + 1] = (
        kf.filter_update(
            filtered_state_means[t],
            filtered_state_covariances[t],
            y_test[t + 1]
        )
    )

plt.figure()

smoothed_state_means, _ = kf.smooth(filtered_state_means)
lines_true = plt.plot(y_test[:, attributes.index(dependent_variable)], color='b')
lines_filt = plt.plot(filtered_state_means[:, attributes.index(dependent_variable)], color='r')
lines_smooth = plt.plot(smoothed_state_means[:, attributes.index(dependent_variable)], color='g')
plt.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
          ('actual', 'kalman', 'smoothed'),
          loc='lower right'
)

plt.show()

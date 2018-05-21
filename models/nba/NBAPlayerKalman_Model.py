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
home = []
for i in range(len(sql['matchup'])):
    #if i == 0:
    #    continue
    if not '@' in sql['matchup'][i]:
        home.append(1.0)
    else:
        home.append(0.0)
#home.append(0.0)
sql['home'] = home
#sql['game_date'] = [i for i in range(len(sql['game_date']))]
#print(sql['game_date'][:10])
sql_test = sql[sql.season_year == test_year]
sql = sql[sql.season_year != test_year]

attributes = ['pts','fg_pct','ast','oreb','stl','fg3_pct']
dependent_variable = 'fg_pct'

x = sql[attributes]
y = sql[[dependent_variable]]
y_test = np.array(sql_test[[dependent_variable]])
x_test = np.array(sql_test[[dependent_variable]])
n_dim_state = len(attributes)

kf = KalmanFilter(n_dim_state=n_dim_state, n_dim_obs=len(attributes))
kf = kf.em(x, y, n_iter=20)
(initial_filtered_state_means, initial_filtered_state_covariances) = kf.filter(x)
(initial_smoothed_state_means, _) = kf.smooth(x)
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
            y_test[t + 1],
            transition_offset=np.array([0])
        )
    )

smoothed_state_means, _ = kf.smooth(filtered_state_means)

filtered_state_means = np.vstack([initial_filtered_state_means, filtered_state_means])
smoothed_state_means = np.vstack([initial_smoothed_state_means, smoothed_state_means])
y = np.vstack([y, y_test])

print("Shape of y: ",y.shape)
print("SHape of y_test: ",y_test.shape)
y = y[-y_test.shape[0]*2+1:, :]
filtered_state_means = filtered_state_means[-y_test.shape[0]*2:, :]
smoothed_state_means = smoothed_state_means[-y_test.shape[0]*2:, :]

print("Shape of y: ",y.shape)
plt.figure()
lines_true = plt.plot(y[:, 0], color='b')
lines_filt = plt.plot(filtered_state_means[:, attributes.index(dependent_variable)], color='r')
lines_smooth = plt.plot(smoothed_state_means[:, attributes.index(dependent_variable)], color='g')
plt.axvline(x=y.shape[0]-y_test.shape[0])
plt.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
          ('actual', 'kalman', 'smoothed'),
          loc='lower right'
)

plt.show()

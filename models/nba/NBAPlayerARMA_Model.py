import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

plt.interactive(False)
conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")

lebron_james = pd.read_sql('select person_id from nba_players_all where display_first_last = \'LeBron James\'', conn)
lebron_james_id = lebron_james['person_id'][0]
print('LeBron James ID: ', lebron_james_id)

# get dataset for lebron james
sql = pd.read_sql('select * from nba_players_game_stats where player_id = \''+lebron_james_id+'\' and game_date is not null and '+
                  'season_type = \'Regular Season\' and min > 0 and fg3_pct is not null '+
                  'and season_year > 2015 '+
                  'order by game_date asc', conn)
test_year = 2017

sql['fga_pm'] = sql['fga']/sql['min']
#sql['game_date'] = [str(date) for date in sql['game_date']]
print(sql['game_date'][:10])
sql_test = sql[sql.season_year == test_year]
#sql = sql[sql.season_year != test_year]

output_attributes = ['fg_pct']

y = np.array(sql[output_attributes], dtype=np.float64).flatten()
dates = np.array(sql['game_date']).flatten()

y_test = np.array(sql_test[output_attributes], dtype=np.float64)
dates_test = np.array(sql_test['game_date']).flatten()

y = pd.Series(y, index=dates)
arma_mod = sm.tsa.ARMA(y, (1, 1))
arma_res = arma_mod.fit(disp=False)
print(arma_res.params)
print(arma_res.summary())

_, ax = plt.subplots(figsize=(10, 8))
fig = arma_res.plot_predict(ax=ax)
df = sql[['game_date', 'fg_pct']]
#ax = df.plot(ax=ax)
fig = arma_res.plot_predict(100, dynamic=True, ax=ax, plot_insample=False)


legend = ax.legend(loc='upper left')
plt.show()

import pandas as pd
from sqlalchemy import create_engine
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
'''
    Runs an OLS regression on the spread (home points - away points) during 
    the Regular Season using the following variables for both home and away teams:
    turnovers (tov), offensive rebounds (oreb), field goal percentage (fg_pct),
    and 3 pointers made (fg3m).
    
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 spread   R-squared:                       0.843
Model:                            OLS   Adj. R-squared:                  0.843
Method:                 Least Squares   F-statistic:                 2.388e+04
Date:                Sun, 20 May 2018   Prob (F-statistic):               0.00
Time:                        10:23:36   Log-Likelihood:            -1.0909e+05
No. Observations:               35585   AIC:                         2.182e+05
Df Residuals:                   35576   BIC:                         2.183e+05
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -1.7239      0.376     -4.582      0.000      -2.461      -0.986
h_tov         -0.8490      0.007   -118.218      0.000      -0.863      -0.835
a_tov          0.8959      0.007    129.907      0.000       0.882       0.909
h_oreb         0.8063      0.007    116.958      0.000       0.793       0.820
a_oreb        -0.7681      0.007   -108.289      0.000      -0.782      -0.754
h_fg_pct     152.3911      0.512    297.763      0.000     151.388     153.394
a_fg_pct    -149.2326      0.526   -283.493      0.000    -150.264    -148.201
h_fg3m         0.8195      0.009     94.768      0.000       0.803       0.836
a_fg3m        -0.8534      0.009    -96.993      0.000      -0.871      -0.836
==============================================================================
Omnibus:                     1164.228   Durbin-Watson:                   1.992
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3276.710
Skew:                          -0.027   Prob(JB):                         0.00
Kurtosis:                       4.486   Cond. No.                         608.
==============================================================================


Runs a second regression to predict the total points (home points + away points)
during the regular season using the following variables:
offensive rebounds (oreb), field goal percentage (fg_pct), three pointers made (f3gm),
and assists (ast).

Note: Condition number is fairly large. Can remove the assists variables (h_ast, a_ast)
to reduce the condition number at the expense of 0.03 to the R-squared, which may be
worth it.

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  total   R-squared:                       0.635
Model:                            OLS   Adj. R-squared:                  0.635
Method:                 Least Squares   F-statistic:                     7490.
Date:                Sun, 20 May 2018   Prob (F-statistic):               0.00
Time:                        10:42:34   Log-Likelihood:            -1.3787e+05
No. Observations:               34469   AIC:                         2.758e+05
Df Residuals:                   34460   BIC:                         2.758e+05
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -10.9622      0.951    -11.530      0.000     -12.826      -9.099
h_oreb         1.1283      0.018     62.238      0.000       1.093       1.164
a_oreb         1.1538      0.019     62.330      0.000       1.117       1.190
h_fg_pct     157.2853      1.665     94.494      0.000     154.023     160.548
a_fg_pct     159.6438      1.624     98.324      0.000     156.461     162.826
h_fg3m         0.8811      0.023     38.196      0.000       0.836       0.926
a_fg3m         1.0368      0.024     43.818      0.000       0.990       1.083
h_ast          0.6506      0.017     38.134      0.000       0.617       0.684
a_ast          0.5754      0.017     33.440      0.000       0.542       0.609
==============================================================================
Omnibus:                     1427.227   Durbin-Watson:                   1.789
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1937.133
Skew:                           0.422   Prob(JB):                         0.00
Kurtosis:                       3.797   Cond. No.                         934.
==============================================================================
'''

team_id = '1610612739'  # cleveland
conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
sql = pd.read_sql('select * from nba_games_all where game_date is not null and ' +
                  'season_type = \'Regular Season\' and h_fg3_pct is ' +
                  'not null and a_fg3_pct is not null and \''+team_id+'\'= ANY(ARRAY[h_team_id]) ' +
                  'and season_year >= 2010 ' +
                  'order by game_date asc', conn)

test_season = 2017

sql['spread'] = sql['h_pts'] - sql['a_pts']
sql['total'] = sql['h_pts'] + sql['a_pts']
home = []
total = []
spread = []
for i in range(len(sql)):
    if i < len(sql)-1:
        total.append(sql['total'][i+1])
        spread.append(sql['spread'][i+1])
    if sql['h_team_id'][i] == team_id:
        home.append(1.0)
    else:
        home.append(0.0)

sql['home'] = home
sql = sql[:-1]
sql['spread'] = spread
sql['total'] = total

test_data = sql[sql.season_year == test_season]
sql = sql[sql.season_year != test_season]

input_attributes = ['h_tov', 'h_oreb',
                    'h_fg_pct', 'h_fg3m',
                    #'home'
                    ]

# model to predict point spread (h_pts - a_pts)
results = smf.ols('spread ~ '+'+'.join(input_attributes), data=sql).fit()
# Inspect the results
print(results.summary())
predictions = results.predict(test_data)
errors = np.array(test_data['spread'])-np.array(predictions)
print('Average error on spread: ', np.mean(np.abs(errors), -1))
# Inspect
plt.figure()
lines_true = plt.plot(errors, color='b')
plt.show()

input_attributes = ['h_oreb',
                    'h_fg_pct',
                    'h_fg3m',
                    'h_ast'
                    ]

# model to predict the total score (h_pts + a_pts)
results = smf.ols('total ~ '+'+'.join(input_attributes), data=sql).fit()
print(results.summary())

predictions = results.predict(test_data)
errors = np.array(test_data['total'])-np.array(predictions)
print('Average error on totals: ', np.mean(np.abs(errors), -1))
# Inspect the results
plt.figure()
lines_true = plt.plot(errors, color='b')
plt.show()

import pandas as pd
from sqlalchemy import create_engine
import statsmodels.formula.api as smf
import numpy as np
from keras.layers import Layer, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.models import Model
import matplotlib.pyplot as plt

team_id = '1610612739'  # cleveland  '1610612757' # portland
conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
sql = pd.read_sql('select * from nba_games_all where game_date is not null and ' +
                  'season_type = \'Regular Season\' and h_fg3_pct is ' +
                  'not null and a_fg3_pct is not null '+
                  'and \'' + team_id + '\'= ANY(ARRAY[h_team_id,a_team_id]) ' +
                  'and season_year >= 2010 ' +
                  'order by game_date asc', conn)

test_season = 2017

sql['spread'] = sql['h_pts'] - sql['a_pts']
sql['total'] = sql['h_pts'] + sql['a_pts']

outcomes = []
home = []
for i in range(len(sql)):
    if i > 0:
        if sql['h_pts'][i] > sql['a_pts'][i]:
            outcomes.append(1.0)
        else:
            outcomes.append(0.0)
    if sql['h_team_id'][i] == team_id:
        home.append(1.0)
    else:
        home.append(0.0)

sql['home'] = home
sql = sql[:-1]  # lag
sql['y'] = outcomes
input_attributes = ['h_tov', 'h_oreb',
                    'h_fg_pct', 'h_fg3m',
                    'spread','total',
                    'h_pts', 'home'
                    ]

sql_test = sql[sql.season_year == test_season]
sql = sql[sql.season_year != test_season]

y = sql['y']
y_test = sql_test['y']

sql = sql[input_attributes]
sql_test = sql_test[input_attributes]

sql = sql.astype(np.float64, copy=False)
sql_test = sql_test.astype(np.float64, copy=False)

y = np.array(y)
y_test = np.array(y_test)

x = np.array(sql)
x_test = np.array(sql_test)

x_mean = np.mean(x, 0)
x_std = np.std(x, 0)

x = (x-x_mean) / x_std
x_test = (x_test - x_mean) / x_std

n_features = len(input_attributes)
X = Input((n_features,))

loss = 'binary_crossentropy'

model = Dense(8, activation='tanh')(X)
model = Dropout(0.5)(model)
model = Dense(1, activation='sigmoid')(model)

model = Model(inputs=X, outputs=model)

model.compile(loss=loss,
              optimizer=Adam(lr=0.001, decay=0.0001),
              metrics=['accuracy'])

model.fit(x, y,
          batch_size=sql.shape[0],
          epochs=1000,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


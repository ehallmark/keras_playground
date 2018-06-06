import pandas as pd
from keras.layers import Dense, Reshape, Concatenate, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sqlalchemy import create_engine
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
'''

'''


def test_model(model, x, y):
    predictions = model.predict(x)
    binary_predictions = (predictions >= 0.5).astype(int)
    binary_errors = np.array(y).flatten() != np.array(binary_predictions).flatten()
    binary_errors = binary_errors.astype(int)
    errors = np.array(np.array(y).flatten() - np.array(predictions).flatten())
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

def get_all_data(attributes, test_season=2017, start_year=2003):
    conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
    sql = pd.read_sql('''
    select p1.*,p2.w as w0, p2.l as l0, p2.plus_minus::double precision/greatest(1,p2.num_games_played) as plus_minus0, p2.fg_pct as fg_pct0, p2.dreb as dreb0, p2.blk as blk0, p2.ast as ast0, p2.pf as pf0,p2.min as min0, p2.fta as fta0, p2.tov as tov0, p2.pts as pts0, p2.oreb as oreb0, p2.fga as fga0, p2.fgm as fgm0, 
                      p3.w as w1, p3.l as l1, p3.plus_minus::double precision/greatest(1,p3.num_games_played) as plus_minus1, p3.fg_pct as fg_pct1, p3.dreb as dreb1, p3.blk as blk1, p3.ast as ast1, p3.pf as pf1, p3.min as min1, p3.fta as fta1, p3.tov as tov1, p3.pts as pts1, p3.oreb as oreb1, p3.fga as fga1, p3.fgm as fgm1, 
                      p4.w as w2, p4.l as l2, p4.plus_minus::double precision/greatest(1,p4.num_games_played) as plus_minus2, p4.fg_pct as fg_pct2, p4.dreb as dreb2, p4.blk as blk2, p4.ast as ast2, p4.pf as pf2, p4.min as min2, p4.fta as fta2, p4.tov as tov2, p4.pts as pts2, p4.oreb as oreb2, p4.fga as fga2, p4.fgm as fgm2, 
                      p5.plus_minus as plus_minus3, p5.fg_pct as fg_pct3, p5.dreb as dreb3, p5.blk as blk3, p5.ast as ast3, p5.pf as pf3, p5.min as min3, p5.fta as fta3, p5.tov as tov3, p5.pts as pts3, p5.oreb as oreb3, p5.fga as fga3, p5.fgm as fgm3, 
                      player.height as height0,
                      player.weight as weight0,
                      case when player.position ilike \'%guard%\' then 1.0 else 0.0 end as guard,
                      case when player.position ilike \'%forward%\' then 1.0 else 0.0 end as forward,
                      case when player.position ilike \'%center%\' then 1.0 else 0.0 end as center,
                      extract (year from p1.game_date) - extract(year from player.birth_date) as exp0
                      from nba_players_game_stats as p1 
                      left join nba_players_game_stats_month_lag as p2 
                      on ((p1.player_id,p1.game_id)=(p2.player_id,p2.game_id)) 
                      left join nba_players_game_stats_previous_matchups as p3
                      on ((p1.player_id,p1.game_id)=(p3.player_id,p3.game_id)) 
                      left join nba_players_all as player
                      on (p1.player_id=player.person_id)
                      left join nba_players_season_stats as p4
                      on ((p1.player_id,p1.game_id)=(p4.player_id,p4.game_id))
                      left join nba_players_season_stats_var as p5
                      on ((p1.player_id,p1.game_id)=(p5.player_id,p5.game_id)) 
                      where p1.game_date is not null and  
                      p1.season_year >= {{START}} and p1.season_year<={{END}} and 
                      p1.season_type = \'Regular Season\' and p1.fg3_pct is not null and 
                      p1.stl is not null and p1.oreb is not null and p1.plus_minus is not null and
                      coalesce(p2.min,0) > 10
                      order by p1.game_date asc
    '''.replace('{{START}}',str(start_year)).replace('{{END}}',str(test_season)), conn)

    wins = [bool_to_int(y == 'W') for y in sql['wl']]
    print("Num wins: " + str(len(wins)))
    print("Num datapoints: ", str(sql.shape[0]))
    sql['y'] = wins
    sql['home'] = [bool_to_int(not '@' in matchup) for matchup in sql['matchup']]
    sql.fillna(inplace=True,value=0.0)
    test_data = sql[sql.season_year == test_season]
    sql = sql[sql.season_year != test_season]
    y = sql['y'].astype(np.float64)
    sql = sql[attributes].astype(np.float64)
    test_y = test_data['y'].astype(np.float64)
    test_data = test_data[attributes].astype(np.float64)
    data = np.array(sql)
    test_data = np.array(test_data)
    return (data, y), (test_data, test_y)

test_season = 2017

input_attributes = [
    'home',
    'dreb0',
    'blk0',
    'ast0',
    'pf0',
    'min0',
    'fta0',
    'plus_minus0',
    'fgm0',
    'fga0',
    'tov0',
    'oreb0',
    'pts0',
    'dreb1',
    'blk1',
    'ast1',
    'pf1',
    'min1',
    'fta1',
    'plus_minus1',
    'fgm1',
    'fga1',
    'tov1',
    'oreb1',
    'pts1',
    'exp0',
    'height0',
    'weight0',
    'blk2',
    'ast2',
    'pf2',
    'min2',
    'fta2',
    'plus_minus2',
    'fgm2',
    'fga2',
    'tov2',
    'oreb2',
    'pts2',
    'blk3',
    'ast3',
    'pf3',
    'min3',
    'fta3',
    'plus_minus3',
    'fgm3',
    'fga3',
    'tov3',
    'oreb3',
    'pts3',
    'w0', 'w1', 'w2',
    'l0', 'l1', 'l2',
    'guard',
    'forward',
    'center'
]

if __name__ == '__main__':

    data, test_data = get_all_data(input_attributes, test_season=test_season)
    print('Data: ', data[0:10])
    def cell(x1,x2, n_units):
        c = Concatenate()([x1,x2])
        c = BatchNormalization()(c)
        c = Dense(n_units, activation='relu')(c)
        c = Dropout(0.5)(c)
        return c

    X = Input((len(input_attributes),))

    hidden_units = 256
    num_cells = 2
    batch_size = 256

    norm = BatchNormalization()(X)
    model1 = Dense(hidden_units, activation='relu')(norm)
    model2 = Dense(hidden_units, activation='relu')(norm)
    for i in range(num_cells):
        model1 = cell(model1,model2,hidden_units)
        model2 = cell(model2,model1,hidden_units)

    model = Dense(1, activation='sigmoid')(model2)
    model = Model(inputs=X, outputs=model)
    model.compile(optimizer=Adam(lr=0.0001, decay=0.0001), loss='mean_squared_error', metrics=['accuracy'])

    model_file = 'nba_player_model_keras_nn.h5'

    prev_accuracy = 0.0
    best_accuracy = 0.0
    for i in range(50):
        model.fit(data[0], data[1], batch_size=batch_size, initial_epoch=i, epochs=i+1, validation_data=test_data, shuffle=True)
        binary_correct, n, binary_percent, avg_error = test_model(model, test_data[0], test_data[1])
        print('Correctly predicted: ' + str(binary_correct) + ' out of ' + str(n) +
              ' (' + to_percentage(binary_percent) + ')')
        print('Average error: ', to_percentage(avg_error))
        if binary_percent > best_accuracy:
            best_accuracy = binary_percent
            # save
            model.save(model_file)
            print('Saved.')
        prev_accuracy = binary_percent



    print(model.summary())

    print('Most recent accuracy: ', prev_accuracy)
    print('Best accuracy: ', best_accuracy)


exit(0)



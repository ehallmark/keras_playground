from keras.layers import Dense, Reshape, Add, Multiply, Recurrent, Concatenate, LSTM, Lambda, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras as k
from models.atp_tennis.TennisMatchOutcomeLogit import load_data, to_percentage
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
import numpy as np
import pandas as pd
import datetime
np.random.seed(23952)


def test_model(model, x, y):
    predictions = model.predict(x)
    avg_error = 0
    avg_error += np.mean(np.abs(np.array(y).flatten() - np.array(predictions).flatten()))
    return avg_error


# previous year quality
input_attributes = list(tennis_model.input_attributes0)

additional_attributes = [
    'y',
    'spread',
    'clay',
    'grass',
    'tournament_rank',
    'round_num',
    #'current_aces',
    #'opp_current_aces',
    #'current_double_faults',
    #'opp_current_double_faults',
    #'current_break_points_made',
    #'opp_current_break_points_made',
    #'current_return_points_won',
    #'opp_current_return_points_won',
    # 'current_return_points_attempted',
    # 'opp_current_return_points_attempted',
    #'current_first_serve_points_made',
    #'opp_current_first_serve_points_made',
    #'current_first_serve_points_attempted',
    #'opp_current_first_serve_points_attempted',
    #'year',
]

for attr in additional_attributes:
    input_attributes.append(attr)

all_attributes = list(input_attributes)
if 'y' not in all_attributes:
    all_attributes.append('y')
if 'spread' not in all_attributes:
    all_attributes.append('spread')
meta_attributes = ['player_id', 'opponent_id', 'tournament', 'round_num', 'year', 'clay',
                   'tournament_rank', 'start_date', 'court_surface']
for meta in meta_attributes:
    if meta not in all_attributes:
        all_attributes.append(meta)


if __name__ == '__main__':
    test_date = datetime.date(2016, 1, 1)
    end_date = datetime.date(2017, 1, 1)
    data = load_data(all_attributes, test_season=end_date.strftime('%Y-%m-%d'), start_year='1990-01-01', masters_min=101)

    # data = data[data.tournament_rank < 101]
    # test_data = test_data[test_data.tournament_rank < 101]

    max_len = 64
    samples = 10000
    test_samples = 1000

    all_players = list(set(data['player_id']))

    data.sort_values(by=['player_id', 'start_date', 'tournament', 'round_num'],
                     inplace=True, ascending=True, kind='mergesort')

    all_matches = data[['player_id', 'opponent_id', 'tournament', 'start_date']].iloc[:]

    data.set_index(['player_id'], inplace=True)

    data_grouped = data.groupby(by=['player_id'], sort=True)

    x = np.zeros((samples, len(input_attributes), max_len))
    x2 = np.zeros((samples, len(input_attributes), max_len))
    x_test = np.zeros((test_samples, len(input_attributes), max_len))
    x2_test = np.zeros((test_samples, len(input_attributes), max_len))
    y = np.zeros((samples,))
    y_test = np.zeros((test_samples,))
    cnt = 0
    indices = list(range(samples))
    test_indices = list(range(test_samples))
    np.random.shuffle(indices)
    np.random.shuffle(test_indices)
    while len(indices) > 0:
        sample = None
        while sample is None or sample.shape[0] < 2:
            match_row = all_matches.iloc[np.random.randint(0,all_matches.shape[0])]
            player = match_row['player_id']
            sample = data_grouped.get_group(player)

        n = max(1,sample.shape[0]-1-max_len)
        for rowIdx in range(n):
            rowIdx = n - 1 - rowIdx
            if cnt % 1000 == 999:
                print('Sample', cnt, 'out of', samples)
            cnt = cnt + 1
            last_idx = 0
            row = sample.iloc[rowIdx]
            opponent = row['opponent_id']
            opp_sample = data_grouped.get_group(opponent)
            opp_sample = opp_sample[((opp_sample.start_date<row['start_date'])|((opp_sample.start_date==row['start_date'])&(opp_sample.tournament==row['tournament'])&(opp_sample.round_num<row['round_num'])))]
            test = row['start_date'] >= test_date
            if test:
                if len(test_indices)==0:
                    continue
                i = test_indices.pop()
            else:
                if len(indices) == 0:
                    continue
                i = indices.pop()

            for j in range(max_len):
                last_idx = rowIdx - j
                opp_index = opp_sample.shape[0]-1-j
                if test:
                    if last_idx >= 0:
                        x_test[i, :, max_len - 1 - j] = np.array(sample.iloc[last_idx][input_attributes])
                    else:
                        x_test[i, :, max_len - 1 - j] = np.array([0.0] * len(input_attributes))

                    if opp_index >= 0:
                        x2_test[i, :, max_len - 1 - j] = np.array(opp_sample.iloc[opp_index][input_attributes])
                    else:
                        x2_test[i, :, max_len - 1 - j] = np.array([0.0] * len(input_attributes))
                else:
                    if last_idx >= 0:
                        x[i, :, max_len - 1 - j] = np.array(sample.iloc[last_idx][input_attributes])
                    else:
                        x[i, :, max_len - 1 - j] = np.array([0.0] * len(input_attributes))

                    if opp_index >= 0:
                        x2[i, :, max_len - 1 - j] = np.array(opp_sample.iloc[opp_index][input_attributes])
                    else:
                        x2[i, :, max_len - 1 - j] = np.array([0.0] * len(input_attributes))

            if test:
                y_test[i] = float(sample['y'].iloc[rowIdx+1])
            else:
                y[i] = float(sample['y'].iloc[rowIdx+1])

    # data = data[data.clay<0.5]
    # test_data = test_data[test_data.clay<0.5]
    X1 = Input((len(input_attributes), max_len))
    X2 = Input((len(input_attributes), max_len))

    data = ([x, x2], y)
    test_data = ([x_test, x2_test], y_test)

    hidden_units = 256
    num_cells = 1
    batch_size = 128
    load_previous = False
    if load_previous:
        model = k.models.load_model('tennis_match_rnn.h5')
        model.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        norm = BatchNormalization()
        model = Concatenate(axis=1)([norm(X1), norm(X2)])

        for i in range(num_cells):
            model = LSTM(hidden_units, return_sequences=i != num_cells-1)(model)

        model = Dense(1, activation='sigmoid')(model)
        model = Model(inputs=[X1, X2], outputs=model)
        model.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    model_file = 'tennis_match_rnn.h5'
    prev_error = None
    best_error = None
    for i in range(50):
        model.fit(data[0], data[1], batch_size=batch_size, initial_epoch=i, epochs=i+1, validation_data=test_data,
                  shuffle=True)
        avg_error = test_model(model, test_data[0], test_data[1])
        print('Average error: ', to_percentage(avg_error))
        if best_error is None or best_error > avg_error:
            best_error = avg_error
            # save
            model.save(model_file)
            print('Saved.')
        prev_error = avg_error

    print(model.summary())
    print('Most recent model error: ', prev_error)
    print('Best model error: ', best_error)


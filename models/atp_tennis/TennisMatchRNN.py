from keras.layers import Dense, Reshape, Add, Multiply, Recurrent, Concatenate, LSTM, Lambda, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras as k
from models.atp_tennis.TennisMatchOutcomeLogit import load_data, to_percentage
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
import numpy as np
import pandas as pd
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
    data = load_data(all_attributes, test_season='2012-01-01', start_year='1995-01-01', masters_min=111)
    test_data = load_data(all_attributes, test_season='2012-12-31', start_year='2012-01-01', masters_min=111)
    # data = data[data.tournament_rank < 101]
    # test_data = test_data[test_data.tournament_rank < 101]
    max_len = 32
    samples = 10000
    test_samples = 1000

    all_players = list(set(data['player_id']))
    test_players = list(set(test_data['player_id']))

    data.sort_values(by=['player_id', 'start_date', 'tournament', 'round_num'],
                     inplace=True, ascending=True, kind='mergesort')
    test_data.sort_values(by=['player_id', 'start_date', 'tournament', 'round_num'],
                          inplace=True, ascending=True, kind='mergesort')
    data.set_index(['player_id'], inplace=True)
    test_data.set_index(['player_id'], inplace=True)
    # print("Federer test data sorted: ", test_data)

    data_grouped = data.groupby(by=['player_id'], sort=True)
    test_data_grouped = test_data.groupby(by=['player_id'], sort=True)

    x = np.zeros((samples, len(input_attributes), max_len))
    x_test = np.zeros((test_samples, len(input_attributes), max_len))
    y = []
    y_test = []
    for n, test in [(samples, False), (test_samples, True)]:
        cnt = 0
        indices = list(range(n))
        np.random.shuffle(indices)
        while len(indices) > 0:
            sample = None
            while sample is None or sample.shape[0] < 2:
                if test:
                    player = test_players[np.random.randint(0, len(test_players))]
                    sample = test_data_grouped.get_group(player)
                else:
                    player = all_players[np.random.randint(0, len(all_players))]
                    sample = data_grouped.get_group(player)

            l = min([max_len, sample.shape[0]-1])
            if l == sample.shape[0]-1:
                start_indices = [0]
            else:
                start_indices = []
                for j in range(1+int(np.log(1+sample.shape[0]-l))):
                    start_indices.append(np.random.randint(0, sample.shape[0]-1 - l))
            for start_idx in start_indices:
                if len(indices)==0:
                    break
                if cnt % 1000 == 999:
                    print('Sample', cnt, 'out of', n)
                cnt = cnt + 1
                last_idx = 0
                i = indices[0]
                indices = indices[1:]
                for j in range(l):
                    if test:
                        x_test[i, :, j] = np.array(sample.iloc[start_idx + j][input_attributes])
                    else:
                        x[i, :, j] = np.array(sample.iloc[start_idx + j][input_attributes])
                    last_idx = start_idx + j

                if test:
                    y_test.append(float(sample['y'].iloc[last_idx+1]))
                else:
                    y.append(float(sample['y'].iloc[last_idx+1]))

    y = np.array(y)
    y_test = np.array(y_test)

    # data = data[data.clay<0.5]
    # test_data = test_data[test_data.clay<0.5]
    X1 = Input((len(input_attributes), max_len))

    data = (x, y)
    test_data = (x_test, y_test)

    hidden_units = 128
    num_cells = 2
    batch_size = 128
    dropout = 0.25
    load_previous = False
    if load_previous:
        model = k.models.load_model('tennis_match_rnn.h5')
        model.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    else:

        norm = BatchNormalization()
        model = norm(X1)
        for i in range(num_cells):
            model = LSTM(hidden_units, return_sequences=i != num_cells-1)(model)

        model = Dense(1, activation='sigmoid')(model)
        model = Model(inputs=X1, outputs=model)
        model.compile(optimizer=Adam(lr=0.001, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])

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


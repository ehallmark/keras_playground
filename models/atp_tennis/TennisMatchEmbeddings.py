from keras.layers import Dense, Reshape, Add, Multiply, Recurrent, Concatenate, LSTM, Lambda, Input, \
    Embedding, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.activations import relu
from keras.initializers import RandomUniform
import keras as k
import keras.backend as K
from models.atp_tennis.TennisMatchBettingSklearnModels import load_data
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
import numpy as np
import pandas as pd
import datetime
np.random.seed(23952)


def bet_loss(epsilon=0):
    def helper(y, y_bin):
        y_bin = K.flatten(y_bin)
        y_odds = K.flatten(y[:, 1])
        y = K.flatten(y[:, 0])
        y_correct = y_bin * y
        y_wrong = y_bin * (1.0 - y)
        y_correct = y_correct / (y_correct + K.epsilon())
        y_wrong = y_wrong / (y_wrong + K.epsilon())
        profit = y_correct / y_odds - y_wrong / (1.0 - y_odds)
        print('Profit', profit)
        return - K.mean(profit, axis=-1)
    return helper


def bet_loss2(epsilon=0):
    def helper(y, y_bin):
        y_bin = K.flatten(y_bin)
        y_odds = K.flatten(y[:, 1])
        y = K.flatten(y[:, 0])
        y_correct = y_bin * y
        y_wrong = y_bin * (1.0 - y)
        profit = y_correct / y_odds - y_wrong / (1.0 - y_odds)
        print('Profit', profit)
        return - K.mean(profit, axis=-1)
    return helper


def bet_loss3(epsilon=0):
    def helper(y, y_bin):
        y_bin = K.flatten(y_bin)
        y_odds = K.flatten(y[:, 1])
        y_bin = K.maximum(y_bin - 0.5, 0.)
        y = K.flatten(y[:, 0])
        y_correct = y_bin * y
        y_wrong = y_bin * (1.0 - y)
        y_correct = y_correct / (y_correct + K.epsilon())
        y_wrong = y_wrong / (y_wrong + K.epsilon())
        profit = y_correct / y_odds - y_wrong / (1.0 - y_odds)
        print('Profit', profit)
        return - K.mean(profit, axis=-1)
    return helper


def bet_loss4(epsilon=0):
    def helper(y, y_bin):
        y_bet = K.flatten(y_bin[:, 1])
        y_bin = K.flatten(y_bin[:, 0])
        y_odds = K.flatten(y[:, 1])
        y = K.flatten(y[:, 0])
        # y_bin = np.maximum(y_bin - y_odds, 0.)
        # y_bin = y_bin / (y_bin + 1e-08)
        y_correct = y_bin * y
        y_wrong = y_bin * (1.0 - y)
        profit = y_bet * y_correct / (y_odds / (1.0 - y_odds)) - y_bet * y_wrong
        invested = y_bet * y_bin
        print('Profit', profit)
        return - K.sum(profit, axis=-1) / (K.sum(invested, axis=-1) + K.epsilon())
    return helper


def test_model(model, x, y, epsilon=0):
    y_bin = model.predict(x)
    y_bet = np.maximum(y_bin[:, 1].flatten(), 0.)
    y_bin = y_bin[:, 0].flatten()
    y_odds = y[:, 1].flatten()
    y = y[:, 0].flatten()
    #y_bin = np.maximum(y_bin - y_odds, 0.)
    #y_bin = y_bin / (y_bin + 1e-08)
    y_correct = y_bin * y
    y_wrong = y_bin * (1.0 - y)
    profit = y_bet * y_correct / (y_odds / (1.0 - y_odds)) - y_bet * y_wrong
    invested = y_bet * y_bin
    print('Profit', profit)
    return - np.sum(profit, axis=-1) / (np.sum(invested, axis=-1) + 1e-08)


# previous year quality
input_attributes = list(tennis_model.input_attributes0)

additional_attributes = [
    'clay',
    'grass',
    #'hard',
    'tournament_rank_percent',
    'round_num_percent',
    'first_round',
]

betting_attributes = [
    'prev_odds',
    'opp_prev_odds',
    'underdog_wins',
    'opp_underdog_wins',
    'fave_wins',
    'opp_fave_wins',
    'overall_odds_avg',
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

for attr in betting_attributes:
    if attr not in input_attributes:
        input_attributes.append(attr)

if __name__ == '__main__':
    num_test_years = 1
    test_date = datetime.date(2017, 1, 1)
    end_date = datetime.date(test_date.year+num_test_years, 1, 1)

    data, test_data = load_data(start_year='2010-01-01', num_test_years=num_test_years, num_test_months=0, test_year=end_date,
                                              models=None, spread_models=None, masters_min=24, attributes=all_attributes)

    # data = data[data.tournament_rank < 101]
    # test_data = test_data[test_data.tournament_rank < 101]

    # max_len = 128
    # test_samples = data[data.start_date>=test_date].shape[0]
    # samples = data.shape[0] - test_samples

    x = np.array(data[input_attributes])  # np.zeros((samples, len(input_attributes)))
    x_test = np.array(test_data[input_attributes])  # np.zeros((test_samples, len(input_attributes)))
    y = np.array(data[['y', 'ml_odds1']])  # np.zeros((samples,))
    y_test = np.array(test_data[['y', 'ml_odds1']])  # np.zeros((test_samples,))
    spread = np.array(data[['beat_spread', 'odds1']])
    spread_test = np.array(test_data[['beat_spread', 'odds1']])
    '''
        cnt = 0
        indices = list(range(samples))
        test_indices = list(range(test_samples))
        np.random.shuffle(indices)
        np.random.shuffle(test_indices)
        while cnt < data.shape[0]:
            sample = data.iloc[cnt]
    
            if cnt % 1000 == 999:
                print('Sample', cnt, 'out of', samples)
            cnt = cnt + 1
            last_idx = 0
    
            test = sample['start_date'] >= test_date
    
            if test:
                if len(test_indices) == 0:
                    continue
                i = test_indices.pop()
            else:
                if len(indices) == 0:
                    continue
                i = indices.pop()
    
            if test:
                y_test[i] = float(sample['y'])
                x_test[i, :] = np.array(sample[input_attributes])
            else:
                y[i] = float(sample['y'])
                x[i, :] = np.array(sample[input_attributes])
    '''

    X = Input((len(input_attributes),))
    data = (x, y)
    test_data = (x_test, y_test)

    hidden_units = 2048
    num_ff_cells = 8
    batch_size = 128
    dropout = 0.1
    load_previous = False
    use_batch_norm = True

    losses = bet_loss4(0)  # [bet_loss(0), bet_loss2(0), bet_loss3(0), bet_loss4(0)]

    if load_previous:
        model = k.models.load_model('tennis_match_embedding.h5')
        model.compile(optimizer=Adam(lr=0.00001, decay=0.01), loss=losses, metrics=[losses])
    else:

        model = X
        if use_batch_norm:
            model = BatchNormalization()(model)
        for l in range(num_ff_cells):
            model = Dense(hidden_units, activation='tanh')(model)

            if use_batch_norm:
                model = BatchNormalization()(model)
            if dropout > 0:
                model = Dropout(dropout)(model)

        bet = Dense(1, activation=lambda x: relu(x, alpha=0., max_value=None))(model)
        model = Dense(1, activation='sigmoid')(model)
        model = Concatenate()([model, bet])
        model = Model(inputs=X, outputs=model)
        model.compile(optimizer=Adam(lr=0.00001, decay=0.001), loss=losses, metrics=[losses])

    model_file = 'tennis_match_embedding.h5'
    prev_error = None
    best_error = None
    model.summary()
    for i in range(50):
        model.fit(data[0], data[1], batch_size=batch_size, initial_epoch=i, epochs=i+1, validation_data=test_data,
                  shuffle=True)
        avg_error = test_model(model, test_data[0], test_data[1])
        print('Average error: ', avg_error)
        if best_error is None or best_error > avg_error:
            best_error = avg_error
            # save
            model.save(model_file)
            print('Saved.')
        prev_error = avg_error

    print(model.summary())
    print('Most recent model error: ', prev_error)
    print('Best model error: ', best_error)


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


def bet_loss4(y, y_bin):
    y_bet = K.maximum(K.flatten(y_bin[:, 1]), 0.)
    y_bin = K.flatten(y_bin[:, 0])
    y_odds = K.flatten(y[:, 1])
    y = K.flatten(y[:, 0])
    y_correct = y_bin * y
    y_wrong = y_bin * (1.0 - y)
    profit = y_bet * y_correct / (y_odds / (1.0 - y_odds)) - y_bet * y_wrong
    invested = y_bet * y_bin
    print('Profit', profit)
    return - K.sum(profit, axis=-1) / (K.sum(invested, axis=-1) + K.epsilon())


def bet_loss4_masked(y, y_bin):
    y_bet = K.maximum(K.flatten(y_bin[:, 1]), 0.)
    y_mask = K.flatten(y[:, 2])   # mask
    y_bin = K.flatten(y_bin[:, 0]) * y_mask
    y_odds = K.flatten(y[:, 1])
    y = K.flatten(y[:, 0])
    y_correct = y_bin * y
    y_wrong = y_bin * (1.0 - y)
    profit = y_bet * y_correct / (y_odds / (1.0 - y_odds)) - y_bet * y_wrong
    invested = y_bet * y_bin
    print('Profit', profit)
    return - K.sum(profit, axis=-1) / (K.sum(invested, axis=-1) + K.epsilon())


def test_model(model, x, y_list):
    y_pred = model.predict(x)
    losses = []
    for i in range(len(y_pred)):
        y_bin = y_pred[i].astype(dtype=np.float32)
        y = y_list[i].astype(dtype=np.float32)
        y_bet = np.maximum(y_bin[:, 1].flatten(), 0.).astype(dtype=np.float32)
        y_bin = y_bin[:, 0].flatten().astype(dtype=np.float32)
        if y.shape[1] == 3:
            y_mask = y[:, 2].flatten().astype(dtype=np.float32)
            y_bin = y_bin * y_mask
        y_odds = y[:, 1].flatten().astype(dtype=np.float32)
        y = y[:, 0].flatten().astype(dtype=np.float32)
        y_correct = y_bin * y
        y_wrong = y_bin * (1.0 - y).astype(dtype=np.float32)
        profit = y_bet * y_correct / (y_odds / (1.0 - y_odds)) - y_bet * y_wrong
        invested = y_bet * y_bin
        print('Profit', profit)
        loss = - np.sum(profit.astype(dtype=np.float32), axis=-1).astype(dtype=np.float32) / (np.sum(invested.astype(dtype=np.float32), axis=-1) + K.epsilon()).astype(dtype=np.float32)
        losses.append(loss)
    return sum(losses)


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
    'ml_odds1',
    'spread1',
    'odds1',
    'odds2',
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
    K.set_epsilon(10e-8)
    #K.set_floatx('float64')

    num_test_years = 1
    test_date = datetime.date(2017, 1, 1)
    end_date = datetime.date(test_date.year+num_test_years, 1, 1)

    data, test_data = load_data(start_year='2010-01-01', num_test_years=num_test_years, num_test_months=0, test_year=end_date,
                                              models=None, spread_models=None, masters_min=26, attributes=all_attributes)

    # data = data[data.tournament_rank < 101]
    # test_data = test_data[test_data.tournament_rank < 101]

    # max_len = 128
    # test_samples = data[data.start_date>=test_date].shape[0]
    # samples = data.shape[0] - test_samples

    # avoid null exceptions
    data['spread1'].fillna(0., inplace=True)
    test_data['spread1'].fillna(0., inplace=True)

    x = np.array(data[input_attributes])  # np.zeros((samples, len(input_attributes)))
    x_test = np.array(test_data[input_attributes])  # np.zeros((test_samples, len(input_attributes)))
    y = np.hstack([np.array(data[['y', 'ml_odds1']]), np.ones((data.shape[0], 1))])  # np.zeros((samples,))
    y_test = np.hstack([np.array(test_data[['y', 'ml_odds1']]), np.ones((test_data.shape[0], 1))])  # np.zeros((test_samples,))
    spread = np.array(data[['beat_spread', 'odds1', 'spread_mask']])
    spread_test = np.array(test_data[['beat_spread', 'odds1', 'spread_mask']])
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
    data = (x, [y, spread])
    test_data = (x_test, [y_test, spread_test])

    hidden_units = 2048
    num_ff_cells = 6
    batch_size = 128
    dropout = 0.20
    load_previous = False
    use_batch_norm = True

    losses = bet_loss4_masked  # [bet_loss(0), bet_loss2(0), bet_loss3(0), bet_loss4(0)]

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

        bet = Dense(hidden_units, activation='tanh')(model)
        bet_spread = Dense(hidden_units, activation='tanh')(model)
        model_spread = Dense(hidden_units, activation='tanh')(model)
        model = Dense(hidden_units, activation='tanh')(model)
        if use_batch_norm:
            model = BatchNormalization()(model)
            bet = BatchNormalization()(bet)
            bet_spread = BatchNormalization()(bet_spread)
            model_spread = BatchNormalization()(model_spread)
        if dropout > 0:
            model = Dropout(dropout)(model)
            bet = Dropout(dropout)(bet)
            bet_spread = Dropout(dropout)(bet_spread)
            model_spread = Dropout(dropout)(model_spread)

        bet_activation = 'tanh'   # lambda x: relu(x, alpha=0., max_value=10.)
        bet = Dense(1, activation=bet_activation)(bet)
        bet_spread = Dense(1, activation=bet_activation)(bet_spread)
        model = Dense(1, activation='sigmoid')(model)
        model_spread = Dense(1, activation='sigmoid')(model_spread)
        model = Concatenate(name='y_output')([model, bet])
        model_spread = Concatenate(name='spread_output')([model_spread, bet_spread])
        model = Model(inputs=X, outputs=[model, model_spread])
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


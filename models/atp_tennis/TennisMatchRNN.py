from keras.layers import Dense, Reshape, Bidirectional, Add, Multiply, Recurrent, Concatenate, LSTM, Lambda, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras as k
import models.atp_tennis.TennisMatchBettingSklearnModels as tennis_model
from models.atp_tennis.TennisMatchOutcomeLogit import load_data, to_percentage, all_attributes
import models.atp_tennis.database.create_match_tables as database
from models.atp_tennis.TennisMatchEmbeddings import bet_loss4_masked
import numpy as np
from keras.activations import relu, sigmoid
from keras.losses import mean_squared_error
import pandas as pd
import datetime
from models.atp_tennis.TennisMatchOutcomeLogit import input_attributes0
np.random.seed(23952)


def convert_to_3d(matrix, num_steps):
    new_x_len = int(matrix.shape[1] / num_steps)
    new_matrix = np.empty((matrix.shape[0], new_x_len, num_steps))
    for i in range(num_steps):
        new_matrix[:, :, num_steps - 1 - i] = matrix[:, i*new_x_len:(i*new_x_len+new_x_len)]
    return new_matrix


def test_model(model, x, y):
    predictions = model.predict(x)
    avg_error = 0
    avg_error += np.mean(np.abs(np.array(y).flatten() - np.array(predictions).flatten()))
    return avg_error


table_creator = database.quarter_tables

# previous year quality
input_attributes = []
opp_input_attributes = []
max_len = table_creator.num_tables

additional_attributes = list(input_attributes0)

additional_attributes += [
    'clay',
    'grass',
    'tournament_rank_percent',
    'round_num_percent',
    'first_round',
]

all_attributes2 = list(additional_attributes)
for attr in all_attributes:
    if attr not in all_attributes2:
        all_attributes2.append(attr)


for i in range(max_len):
    for attr in table_creator.attribute_names_for(i, include_opp=False):
        input_attributes.append(attr)
    for attr in table_creator.attribute_names_for(i, include_opp=True, opp_only=True):
        opp_input_attributes.append(attr)

if __name__ == '__main__':
    use_sql = True

    if use_sql:
        num_test_years = 1
        test_date = datetime.date(2017, 1, 1)
        end_date = datetime.date(test_date.year+num_test_years, 1, 1)
        start_date = datetime.date(2010, 1, 1)
        data, test_data = tennis_model.load_data(start_year=start_date.strftime('%Y-%m-%d'), num_test_years=num_test_years, num_test_months=0, test_year=end_date,
                                                  models=None, spread_models=None, masters_min=24, attributes=all_attributes2)

        data2 = table_creator.load_data(date=start_date, end_date=end_date, include_null=False)
        print("Merging...")
        data = pd.DataFrame.merge(
            data,
            data2,
            'inner',
            left_on=['start_date', 'player_id', 'opponent_id', 'tournament'],
            right_on=['start_date', 'player_id', 'opponent_id', 'tournament'],
            validate='m:1'
        )
        print("Merging test data...")
        test_data = pd.DataFrame.merge(
            test_data,
            data2,
            'inner',
            left_on=['start_date', 'player_id', 'opponent_id', 'tournament'],
            right_on=['start_date', 'player_id', 'opponent_id', 'tournament'],
            validate='m:1'
        )
        print('Loaded data...')

        def bool_to_int(b):
            if b:
                return 1.0
            else:
                return 0.0

        test_data['ml_mask'] = [bool_to_int(np.isfinite(test_data['ml_odds1'].iloc[i])) for i in range(test_data.shape[0])]
        data['ml_mask'] = [bool_to_int(np.isfinite(data['ml_odds1'].iloc[i])) for i in range(data.shape[0])]

        test_data.to_hdf('test_fixed.hdf', 'test', mode='w')
        data.to_hdf('data_fixed.hdf', 'data', mode='w')
        exit(0)
    else:
        print("Loading data from hdf...")
        data = pd.read_hdf('data_fixed.hdf', 'data')
        print("Loading test data from hdf...")
        test_data = pd.read_hdf('test_fixed.hdf', 'test')

    losses = [mean_squared_error, bet_loss4_masked, bet_loss4_masked]

    data.fillna(value=0., inplace=True)
    test_data.fillna(value=0., inplace=True)

    x = np.array(data[input_attributes])
    x_test = np.array(test_data[input_attributes])
    x2 = np.array(data[opp_input_attributes])
    x2_test = np.array(test_data[opp_input_attributes])
    x3 = np.array(data[additional_attributes])
    x3_test = np.array(test_data[additional_attributes])

    print('Converting data for RNN')
    x = convert_to_3d(x, max_len)
    x_test = convert_to_3d(x_test, max_len)
    x2 = convert_to_3d(x2, max_len)
    x2_test = convert_to_3d(x2_test, max_len)

    X1 = Input((int(len(input_attributes)/max_len), max_len))
    X2 = Input((int(len(opp_input_attributes)/max_len), max_len))
    X3 = Input((len(additional_attributes),))

    y = np.array(data['y'])
    y_test = np.array(test_data['y'])
    ml = np.array(data[['y', 'ml_odds1', 'ml_mask']])  # np.zeros((samples,))
    ml_test = np.array(test_data[['y', 'ml_odds1', 'ml_mask']])  # np.zeros((test_samples,))
    spread = np.array(data[['beat_spread', 'odds1', 'spread_mask']])
    spread_test = np.array(test_data[['beat_spread', 'odds1', 'spread_mask']])

    data = ([x, x2, x3], [y, ml, spread])
    test_data = ([x_test, x2_test, x3_test], [y_test, ml_test, spread_test])


    hidden_units = 256
    hidden_units_ff = 1024
    num_rnn_cells = 2
    num_ff_cells = 4
    batch_size = 128
    dropout = 0
    load_previous = False
    use_batch_norm = True

    if load_previous:
        model = k.models.load_model('tennis_match_rnn.h5')
        model.compile(optimizer=Adam(lr=0.00001, decay=0.001), loss=losses, metrics=losses)
    else:
        if use_batch_norm:
            norm = BatchNormalization()
            model1 = norm(X1)
            model2 = norm(X2)
            model3 = BatchNormalization()(X3)
        else:
            model1 = X1
            model2 = X2
            model3 = X3

        for i in range(num_rnn_cells):
            lstm = Bidirectional(LSTM(hidden_units, activation='tanh', return_sequences=i != num_rnn_cells-1))
            model1 = lstm(model1)
            model2 = lstm(model2)
            model3 = Dense(hidden_units, activation='tanh')(model3)

            if use_batch_norm:
                norm = BatchNormalization()
                model1 = norm(model1)
                model2 = norm(model2)
                model3 = BatchNormalization()(model3)

            if dropout > 0:
                dropout_layer = Dropout(dropout)
                model1 = dropout_layer(model1)
                model2 = dropout_layer(model2)
                model3 = Dropout(dropout)(model3)

        model = Concatenate()([model1, model2, model3])
        model_spread = model
        bet = model 
        bet_spread = model
        for l in range(num_ff_cells):
            model = Dense(hidden_units_ff, activation='tanh')(model)
            model_spread = Dense(hidden_units_ff, activation='tanh')(model_spread)
            bet = Dense(hidden_units_ff, activation='tanh')(bet)
            bet_spread = Dense(hidden_units_ff, activation='tanh')(bet_spread)

            if use_batch_norm:
                model = BatchNormalization()(model)
                model_spread = BatchNormalization()(model_spread)
                bet = BatchNormalization()(bet)
                bet_spread = BatchNormalization()(bet_spread)

            if dropout > 0:
                model = Dropout(dropout)(model)
                model_spread = Dropout(dropout)(model_spread)
                bet = Dropout(dropout)(bet)
                bet_spread = Dropout(dropout)(bet_spread)

        outcome = Dense(hidden_units, activation='tanh')(model)
        outcome = Dense(1, activation='sigmoid')(outcome)

        def bet_activation(x_):
            return relu(x_, alpha=0., max_value=None)

        bet = Dense(1, activation=bet_activation)(bet)
        bet_spread = Dense(1, activation=bet_activation)(bet_spread)
        model = Dense(1, activation='sigmoid')(model)
        model_spread = Dense(1, activation='sigmoid')(model_spread)
        model = Concatenate(name='y_output')([model, bet])
        model_spread = Concatenate(name='spread_output')([model_spread, bet_spread])
        model = Model(inputs=[X1, X2, X3], outputs=[outcome, model, model_spread])
        model.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss=losses, metrics=[])

    model_file = 'tennis_match_rnn.h5'
    prev_error = None
    best_error = None
    for i in range(100):
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


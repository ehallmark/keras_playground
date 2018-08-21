print("Loaded classes...")
from keras.layers import Dense, Reshape, Bidirectional, Add, Multiply, Recurrent, Concatenate, LSTM, Lambda, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras as k
import keras.backend as K
import models.atp_tennis.TennisMatchBettingSklearnModels as tennis_model
print("Loaded classes...")
from models.atp_tennis.TennisMatchOutcomeLogit import load_data, to_percentage, all_attributes
import models.atp_tennis.database.create_match_tables as database
import models.atp_tennis.database.tournament_tables as tourney_database
import models.atp_tennis.database.daily_tables as daily_database
from models.atp_tennis.TennisMatchEmbeddings import bet_loss4_masked
import sklearn.metrics as metrics
import numpy as np
print("Loaded classes...")

from keras.activations import relu, sigmoid
from keras.losses import mean_squared_error, categorical_crossentropy
import pandas as pd
import datetime
from models.atp_tennis.TennisMatchOutcomeLogit import input_attributes0
np.random.seed(23952)
print("Loaded classes...")

def convert_to_3d(matrix, num_steps):
    new_x_len = int(matrix.shape[1] / num_steps)
    new_matrix = np.empty((matrix.shape[0], new_x_len, num_steps))
    for i in range(num_steps):
        new_matrix[:, :, num_steps - 1 - i] = matrix[:, i*new_x_len:(i*new_x_len+new_x_len)]
    return new_matrix


def test_model(model, x, y_list):
    y_pred = model.predict(x)
    losses = []
    for i in range(len(y_pred)):
        if i % 2 == 0:
            losses.append(metrics.mean_squared_error(y_list[i], y_pred[i]) * 2.0 ** int(i/2))
        else:
            losses.append(metrics.log_loss(y_list[i], y_pred[i]) * (2.0 ** int(i/2)) / 12.0)

    '''
    for i in range(len(y_pred)-2, len(y_pred)):
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
        profit = y_bet * y_correct / ((y_odds+K.epsilon()) / (1.0 - y_odds+K.epsilon())) - y_bet * y_wrong
        invested = y_bet * y_bin
        print('Profit', profit)
        loss = - np.sum(profit.astype(dtype=np.float32), axis=-1).astype(dtype=np.float32) / (np.sum(invested.astype(dtype=np.float32), axis=-1) + K.epsilon()).astype(dtype=np.float32)
        losses.append(loss)
    '''
    return sum(losses)


quarter_tables = database.quarter_tables
pro_tables = database.pro_tables
junior_tables = database.junior_tables
itf_tables = database.itf_tables
challenger_tables = database.challenger_tables
tournament_tables = tourney_database.tourney_tables
day_tables = daily_database.daily_tables

# previous year quality
input_attributes = []
opp_input_attributes = []
input_attributes2 = []
opp_input_attributes2 = []
input_attributes3 = []
opp_input_attributes3 = []
input_attributes4 = []
opp_input_attributes4 = []
max_len = quarter_tables.num_tables
max_len2 = tournament_tables.num_tables
max_len3 = day_tables.num_tables
print("Max len2:", max_len2)

additional_attributes = list(input_attributes0)

additional_attributes += [
    'clay',
    'grass',
    'tournament_rank_percent',
    'round_num_percent',
    'first_round',
    'avg_ml_estimate',
    'true_prediction',
    'is_qualifier'
]

all_attributes2 = list(additional_attributes)
for attr in all_attributes:
    if attr not in all_attributes2:
        all_attributes2.append(attr)


additional_attributes += [
    #'spread1',
    #'overall_odds_avg',
    #'prev_odds',
    #'opp_prev_odds',
    #'underdog_wins',
    #'opp_underdog_wins',
    #'fave_wins',
    #'opp_fave_wins',
    # 'ml_odds_avg',
]

probability_attrs = [
    #'overall_odds_avg',
    #'prev_odds',
    #'opp_prev_odds',
]

for i in range(max_len):
    for attr in quarter_tables.attribute_names_for(i, include_opp=False):
        input_attributes.append(attr)
    for attr in quarter_tables.attribute_names_for(i, include_opp=True, opp_only=True):
        opp_input_attributes.append(attr)

for table in [junior_tables, itf_tables, challenger_tables, pro_tables]:
    for attr in table.attribute_names_for(0, include_opp=False):
        input_attributes2.append(attr)
    for attr in table.attribute_names_for(0, include_opp=True, opp_only=True):
        opp_input_attributes2.append(attr)

for i in range(max_len2):
    for attr in tournament_tables.attribute_names_for(i, include_opp=False):
        input_attributes3.append(attr)
    for attr in tournament_tables.attribute_names_for(i, include_opp=True, opp_only=True):
        opp_input_attributes3.append(attr)

for i in range(max_len3):
    for attr in day_tables.attribute_names_for(i, include_opp=False):
        input_attributes4.append(attr)
    for attr in day_tables.attribute_names_for(i, include_opp=True, opp_only=True):
        opp_input_attributes4.append(attr)

if __name__ == '__main__':
    use_sql = False
    reload_sql = False

    dataset_name = 'all_data.hdf'  # 'data_fixed.hdf'
    test_dataset_name = 'all_test_data.hdf'  # 'test_fixed.hdf'
    if use_sql:
        num_test_years = 1
        test_date = datetime.date(2016, 1, 1)
        end_date = datetime.date(test_date.year+num_test_years, 1, 1)
        start_date = datetime.date(1995, 1, 1)

        data = load_data(all_attributes2, end_date.strftime('%Y-%m-%d'), start_date.strftime('%Y-%m-%d'),
                         keep_nulls=False, masters_min=1, save=reload_sql, reload=not reload_sql)
        if reload_sql:
            exit(0)

        data2 = quarter_tables.load_data(date=start_date, end_date=end_date, include_null=False)
        data3 = junior_tables.load_data(date=start_date, end_date=end_date, include_null=False)
        data4 = itf_tables.load_data(date=start_date, end_date=end_date, include_null=False)
        data5 = challenger_tables.load_data(date=start_date, end_date=end_date, include_null=False)
        data6 = pro_tables.load_data(date=start_date, end_date=end_date, include_null=False)
        data7 = tournament_tables.load_data(date=start_date, end_date=end_date, include_null=False)
        data8 = day_tables.load_data(date=start_date, end_date=end_date, include_null=False)

        print("column labels2: " + ",".join(list(data2.columns.values)))
        print("column labels3: " + ",".join(list(data3.columns.values)))
        print("column labels4: " + ",".join(list(data4.columns.values)))
        print("column labels5: " + ",".join(list(data5.columns.values)))

        test_data = data[data.start_date >= test_date]
        data = data[data.start_date < test_date]

        print("Merging...")
        for other_data in [data2, data3, data4, data5, data6, data7, data8]:
            if 'player_victory' in list(other_data.columns.values):
                other_data.drop(columns=['player_victory'], inplace=True)
            data = pd.DataFrame.merge(
                data,
                other_data,
                'inner',
                left_on=['start_date', 'player_id', 'opponent_id', 'tournament'],
                right_on=['start_date', 'player_id', 'opponent_id', 'tournament'],
                validate='1:1'
            )
            print("Merging test data...")
            test_data = pd.DataFrame.merge(
                test_data,
                other_data,
                'inner',
                left_on=['start_date', 'player_id', 'opponent_id', 'tournament'],
                right_on=['start_date', 'player_id', 'opponent_id', 'tournament'],
                validate='1:1'
            )
            all_labels = list(data.columns.values)
            labels_set = set(all_labels)
            for label in labels_set:
                if all_labels.count(label) > 1:
                    print("DUPLICATE LABEL FOUND:", label)
        print('Loaded data...')

        def bool_to_int(b):
            if b:
                return 1.0
            else:
                return 0.0

        #test_data['ml_mask'] = [bool_to_int(np.isfinite(test_data['ml_odds1'].iloc[i])) for i in range(test_data.shape[0])]
        #data['ml_mask'] = [bool_to_int(np.isfinite(data['ml_odds1'].iloc[i])) for i in range(data.shape[0])]
        all_labels = list(data.columns.values)
        labels_set = set(all_labels)
        for label in labels_set:
            if all_labels.count(label) > 1:
                print("DUPLICATE LABEL FOUND:", label)

        test_data.to_hdf(test_dataset_name, 'test', mode='w')
        data.to_hdf(dataset_name, 'data', mode='w')
        exit(0)
    else:
        print("Loading data from hdf...")
        data = pd.read_hdf(dataset_name, 'data')
        print("Loading test data from hdf...")
        test_data = pd.read_hdf(test_dataset_name, 'test')
        #data = data[np.isfinite(data.price1)]
        #test_data = test_data[np.isfinite(test_data.price1)]

    losses = [
    #    bet_loss4_masked,
    #    bet_loss4_masked
    ]
    print("column labels: " + ",".join(list(data.columns.values)))

    for prob_attr in probability_attrs:
        # fill probability attributes with 0.5 default value
        data[prob_attr].fillna(value=0.5, inplace=True)
        test_data[prob_attr].fillna(value=0.5, inplace=True)
    data.fillna(value=0., inplace=True)
    test_data.fillna(value=0., inplace=True)

    x = np.array(data[input_attributes])
    x_test = np.array(test_data[input_attributes])
    x2 = np.array(data[opp_input_attributes])
    x2_test = np.array(test_data[opp_input_attributes])
    x3 = np.array(data[input_attributes2])
    x3_test = np.array(test_data[input_attributes2])
    x4 = np.array(data[opp_input_attributes2])
    x4_test = np.array(test_data[opp_input_attributes2])
    x5 = np.array(data[input_attributes3])
    x5_test = np.array(test_data[input_attributes3])
    x6 = np.array(data[opp_input_attributes3])
    x6_test = np.array(test_data[opp_input_attributes3])
    x7 = np.array(data[input_attributes4])
    x7_test = np.array(test_data[input_attributes4])
    x8 = np.array(data[opp_input_attributes4])
    x8_test = np.array(test_data[opp_input_attributes4])
    x9 = np.array(data[additional_attributes])
    x9_test = np.array(test_data[additional_attributes])

    print('Converting data for RNN')
    # quarterly
    x = convert_to_3d(x, max_len)
    x_test = convert_to_3d(x_test, max_len)
    x2 = convert_to_3d(x2, max_len)
    x2_test = convert_to_3d(x2_test, max_len)

    # by tourney
    x5 = convert_to_3d(x5, max_len2)
    x5_test = convert_to_3d(x5_test, max_len2)
    x6 = convert_to_3d(x6, max_len2)
    x6_test = convert_to_3d(x6_test, max_len2)

    # daily
    x7 = convert_to_3d(x7, max_len3)
    x7_test = convert_to_3d(x7_test, max_len3)
    x8 = convert_to_3d(x8, max_len3)
    x8_test = convert_to_3d(x8_test, max_len3)

    X1 = Input((int(len(input_attributes)/max_len), max_len))
    X2 = Input((int(len(opp_input_attributes)/max_len), max_len))
    X3 = Input((len(input_attributes2),))
    X4 = Input((len(opp_input_attributes2),))
    X5 = Input((int(len(input_attributes3)/max_len2), max_len2))
    X6 = Input((int(len(opp_input_attributes3)/max_len2), max_len2))
    X7 = Input((int(len(input_attributes4) / max_len3), max_len3))
    X8 = Input((int(len(opp_input_attributes4) / max_len3), max_len3))
    X9 = Input((len(additional_attributes),))

    spread_range = range(-18, 19, 1)
    num_possible_spreads = len(spread_range)
    print('Num spreads: ', num_possible_spreads)
    y = np.array(data['y'])
    y_test = np.array(test_data['y'])
    np_spread = np.array(data['spread'])
    np_spread_test = np.array(test_data['spread'])
    actual_spread = np.empty((data.shape[0], num_possible_spreads))
    actual_spread_test = np.empty((test_data.shape[0], num_possible_spreads))
    print('calculating spreads...')
    for i in spread_range:
        actual_spread[:, int(num_possible_spreads/2) + i] = (np_spread == i).astype(dtype=np.float32)
        actual_spread_test[:, int(num_possible_spreads/2) + i] = (np_spread_test == i).astype(dtype=np.float32)
    print('done')

    data = ([x, x2, x3, x4, x5, x6, x7, x8, x9], [])
    test_data = ([x_test, x2_test, x3_test, x4_test, x5_test, x6_test, x7_test, x8_test, x9_test], [])

    #print('Test_data: ', test_data[0:10])

    hidden_units = 128
    hidden_units_ff = 128
    num_rnn_cells = 1
    num_ff_cells = 8
    batch_size = 256
    predict_every_n = 4
    dropout = 0.1
    load_previous = False
    use_batch_norm = True

    loss_weights = {}

    c = 0
    for i in range(num_ff_cells):
        if i % predict_every_n == predict_every_n-1:
            data[1].append(y)
            data[1].append(actual_spread)
            test_data[1].append(y_test)
            test_data[1].append(actual_spread_test)
            losses.append(mean_squared_error)
            losses.append(categorical_crossentropy)
            loss_weights['outcome'+str(i)] = 2.0**c
            loss_weights['spread'+str(i)] = (2.0**c) / 12.0
            c += 1

    if load_previous:
        model = k.models.load_model('tennis_match_rnn.h5')
        model.compile(optimizer=Adam(lr=0.00001, decay=0.001), loss_weights=loss_weights, loss=losses, metrics=losses)
    else:
        if use_batch_norm:
            norm = BatchNormalization()
            norm2 = BatchNormalization()
            norm3 = BatchNormalization()
            norm4 = BatchNormalization()
            model1 = norm(X1)
            model2 = norm(X2)
            model3 = norm2(X3)
            model4 = norm2(X4)
            model5 = norm3(X5)
            model6 = norm3(X6)
            model7 = norm4(X7)
            model8 = norm4(X8)
            model9 = BatchNormalization()(X9)
        else:
            model1 = X1
            model2 = X2
            model3 = X3
            model4 = X4
            model5 = X5
            model6 = X6
            model7 = X7
            model8 = X8
            model9 = X9

        for i in range(num_rnn_cells):
            lstm = Bidirectional(LSTM(hidden_units, activation='tanh', return_sequences=i != num_rnn_cells-1))
            lstm2 = Bidirectional(LSTM(hidden_units, activation='tanh', return_sequences=i != num_rnn_cells-1))
            lstm3 = Bidirectional(LSTM(hidden_units, activation='tanh', return_sequences=i != num_rnn_cells-1))
            dense = Dense(hidden_units, activation='tanh')
            model1 = lstm(model1)
            model2 = lstm(model2)
            model3 = dense(model3)
            model4 = dense(model4)
            model5 = lstm2(model5)
            model6 = lstm2(model6)
            model7 = lstm3(model7)
            model8 = lstm3(model8)
            model9 = Dense(hidden_units, activation='tanh')(model9)

            if use_batch_norm:
                norm = BatchNormalization()
                model1 = norm(model1)
                model2 = norm(model2)
                norm2 = BatchNormalization()
                model3 = norm2(model3)
                model4 = norm2(model4)
                norm3 = BatchNormalization()
                model5 = norm3(model5)
                model6 = norm3(model6)
                norm4 = BatchNormalization()
                model7 = norm4(model7)
                model8 = norm4(model8)
                model9 = BatchNormalization()(model9)

            if dropout > 0:
                model1 = Dropout(dropout)(model1)
                model2 = Dropout(dropout)(model2)
                model3 = Dropout(dropout)(model3)
                model4 = Dropout(dropout)(model4)
                model5 = Dropout(dropout)(model5)
                model6 = Dropout(dropout)(model6)
                model7 = Dropout(dropout)(model7)
                model8 = Dropout(dropout)(model8)
                model9 = Dropout(dropout)(model9)

        model = Concatenate()([model1, model2, model3, model4, model5, model6, model7, model8, model9])
        outcomes = []
        for l in range(num_ff_cells):
            prev_model = model
            model = Dense(hidden_units_ff*(max(1, 9 - l)), activation='tanh')(model)
            model = Concatenate()([prev_model, model])

            if use_batch_norm:
                model = BatchNormalization()(model)

            if dropout > 0:
                model = Dropout(dropout)(model)

            if l % predict_every_n == predict_every_n-1:
                outcome = Dense(1, activation='sigmoid', name='outcome' + str(l))(model)
                spread_model = Dense(num_possible_spreads, activation='softmax', name='spread' + str(l))(model)
                outcomes.append(outcome)
                outcomes.append(spread_model)

        model = Model(inputs=[X1, X2, X3, X4, X5, X6, X7, X8, X9], outputs=outcomes)
        model.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss_weights=loss_weights, loss=losses, metrics=[])

    model.summary()
    model_file = 'tennis_match_rnn.h5'
    prev_error = None
    best_error = None
    for i in range(10):
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


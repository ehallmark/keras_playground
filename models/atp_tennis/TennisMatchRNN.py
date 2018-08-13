from keras.layers import Dense, Reshape, Bidirectional, Add, Multiply, Recurrent, Concatenate, LSTM, Lambda, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras as k
from models.atp_tennis.TennisMatchOutcomeLogit import load_data, to_percentage
import models.atp_tennis.database.create_match_tables as database
import numpy as np
import datetime
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


# previous year quality
input_attributes = []
opp_input_attributes = []
max_len = database.num_prior_quarters

for i in range(max_len):
    for attr in database.attribute_names_for(i, include_opp=False):
        input_attributes.append(attr)
    for attr in database.attribute_names_for(i, include_opp=True, opp_only=True):
        opp_input_attributes.append(attr)

if __name__ == '__main__':
    test_date = datetime.date(2011, 1, 1)
    end_date = datetime.date(2012, 1, 1)
    data = database.load_data(date='1995-01-01', include_null=False)

    test_data = data[data.start_date >= test_date]
    test_data = test_data[test_data.start_date <= end_date]
    data = data[data.start_date < test_date]

    x = np.array(data[input_attributes])
    x_test = np.array(test_data[input_attributes])
    x2 = np.array(data[opp_input_attributes])
    x2_test = np.array(test_data[opp_input_attributes])

    x = convert_to_3d(x, max_len)
    x_test = convert_to_3d(x_test, max_len)
    x2 = convert_to_3d(x2, max_len)
    x2_test = convert_to_3d(x2_test, max_len)

    y = np.array(data['y'])
    y_test = np.array(test_data['y'])

    X1 = Input((int(len(input_attributes)/max_len), max_len))
    X2 = Input((int(len(opp_input_attributes)/max_len), max_len))

    data = ([x, x2], y)
    test_data = ([x_test, x2_test], y_test)

    hidden_units = 256
    hidden_units_ff = 1024
    num_rnn_cells = 2
    num_ff_cells = 2
    batch_size = 128
    dropout = 0.1
    load_previous = False
    use_batch_norm = True

    if load_previous:
        model = k.models.load_model('tennis_match_rnn.h5')
        model.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        if use_batch_norm:
            norm = BatchNormalization()
            model1 = norm(X1)
            model2 = norm(X2)
        else:
            model1 = X1
            model2 = X2

        for i in range(num_rnn_cells):
            lstm = Bidirectional(LSTM(hidden_units, activation='tanh', return_sequences=i != num_rnn_cells-1))
            model1 = lstm(model1)
            model2 = lstm(model2)

            if use_batch_norm:
                norm = BatchNormalization()
                model1 = norm(model1)
                model2 = norm(model2)

            if dropout > 0:
                dropout_layer = Dropout(dropout)
                model1 = dropout_layer(model1)
                model2 = dropout_layer(model2)

        model = Lambda(lambda inputs: inputs[0] - inputs[1])([model1, model2])
        if use_batch_norm:
            model = BatchNormalization()(model)

        for l in range(num_ff_cells):
            model = Dense(hidden_units_ff, activation='tanh')(model)

            if use_batch_norm:
                model = BatchNormalization()(model)

            if dropout > 0:
                dropout = Dropout(dropout)(model)

        model = Dense(1, activation='sigmoid')(model)
        model = Model(inputs=X1, outputs=model)
        model.compile(optimizer=Adam(lr=0.00001, decay=0.001), loss='binary_crossentropy', metrics=['accuracy'])


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


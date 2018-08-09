from keras.layers import Dense, Reshape, Add, Multiply, Recurrent, Concatenate, LSTM, Lambda, Input, \
    Embedding, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomUniform
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
    'clay',
    'grass',
    'tournament_rank_percent',
    'round_num_percent',
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
    end_date = datetime.date(2018, 1, 1)
    #min_train_date = datetime.date(1995, 1, 1)
    data = load_data(all_attributes, test_season=end_date.strftime('%Y-%m-%d'), start_year='1995-01-01', masters_min=111)

    # data = data[data.tournament_rank < 101]
    # test_data = test_data[test_data.tournament_rank < 101]

    # max_len = 128
    # test_samples = data[data.start_date>=test_date].shape[0]
    # samples = data.shape[0] - test_samples

    test_data = data[data.start_date >= test_date]
    data = data[data.start_date < test_date]

    x = np.array(data[input_attributes])  # np.zeros((samples, len(input_attributes)))
    x_test = np.array(test_data[input_attributes])  # np.zeros((test_samples, len(input_attributes)))
    y = np.array(data['y'])  # np.zeros((samples,))
    y_test = np.array(test_data['y'])  # np.zeros((test_samples,))
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

    hidden_units = 256
    num_ff_cells = 8
    batch_size = 128
    dropout = 0.5
    load_previous = False
    use_batch_norm = True
    if load_previous:
        model = k.models.load_model('tennis_match_embedding.h5')
        model.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])
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
        model = Dense(1, activation='sigmoid')(model)
        model = Model(inputs=X, outputs=model)
        model.compile(optimizer=Adam(lr=0.0005 , decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    model_file = 'tennis_match_embedding.h5'
    prev_error = None
    best_error = None
    model.summary()
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


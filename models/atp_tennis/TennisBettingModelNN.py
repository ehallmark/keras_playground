import numpy as np
import pandas as pd
import datetime
from random import shuffle
import statsmodels.formula.api as smf
from models.simulation.Simulate import simulate_money_line
import models.atp_tennis.TennisMatchBettingSklearnModels as tennis_model
from models.atp_tennis.TennisMatchOutcomeSklearnModels import load_spread_model, load_outcome_model
from models.atp_tennis.TennisMatchOutcomeLogit import to_percentage
from keras.layers import Dense, Reshape, Add, Multiply, Concatenate, Lambda, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras as k


def test_model(model, x, y):
    predictions = model.predict(x)
    avg_error = 0
    avg_error += np.mean(np.abs(np.array(y).flatten() - np.array(predictions).flatten()))
    return avg_error


models = tennis_model.models

future_matches_only = False
test_year = datetime.date.today()  # IMPORTANT!!

data, test_data = tennis_model.load_data(start_year=tennis_model.start_year, num_test_years=1, test_year=test_year,
                                         models=models, spread_models=None)

input_attributes = list(tennis_model.betting_input_attributes)
for attr in ['clay','grass','grand_slam','overall_odds_avg',
             'spread_odds_avg']:
    if attr not in input_attributes:
        input_attributes.append(attr)

print('Test data shape: ', test_data.shape)

data = data[data.beat_spread.apply(lambda x: np.isfinite(x))]
test_data = test_data[test_data.beat_spread.apply(lambda x: np.isfinite(x))]

X1 = Input((len(input_attributes),))

data = (np.array(data[input_attributes]), np.array(data['beat_spread']))
test_data = (np.array(test_data[input_attributes]), np.array(test_data['beat_spread']))

hidden_units = 128  # len(input_attributes)*2
num_cells = 8
batch_size = 128
dropout = 0.25
load_previous = False
if load_previous:
    model = k.models.load_model('tennis_match_keras_nn_v5.h5')
    model.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])
else:
    def cell(x1, x2, n_units, dropout=0.25):
        batch_norm = BatchNormalization()
        dense = Dense(n_units, activation='tanh')
        concat = Concatenate()
        dropout_layer = Dropout(dropout)
        norm1 = concat([x1, x2])
        norm1 = dense(norm1)
        norm1 = batch_norm(norm1)
        norm1 = dropout_layer(norm1)
        return x2, norm1


    norm = BatchNormalization()
    model1 = norm(X1)
    model2 = Dense(hidden_units, activation='tanh')(model1)
    for i in range(num_cells):
        model1, model2 = cell(model1, model2, hidden_units)

    out1 = Dense(1, activation='sigmoid')(model1)
    model = Model(inputs=X1, outputs=out1)
    model.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])

model_file = 'tennis_match_keras_nn_v5.h5'
prev_error = None
best_error = None
for i in range(50):
    model.fit(data[0], data[1], batch_size=batch_size, initial_epoch=i, epochs=i + 1, validation_data=test_data,
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


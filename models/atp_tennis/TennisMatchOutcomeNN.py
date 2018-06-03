from keras.layers import Dense, Reshape, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from models.atp_tennis.TennisMatchOutcomeLogit import load_data, to_percentage
import numpy as np


def test_model(model, x, y):
    predictions = model.predict(x).flatten()
    binary_predictions = (predictions >= 0.5).astype(int)
    binary_errors = np.array(y) != np.array(binary_predictions)
    binary_errors = binary_errors.astype(int)
    errors = np.array(y - np.array(predictions))
    binary_correct = predictions.shape[0] - int(binary_errors.sum())
    binary_percent = float(binary_correct) / predictions.shape[0]
    avg_error = np.mean(np.abs(errors), -1)
    return binary_correct, y.shape[0], binary_percent, avg_error


input_attributes = [
    'mean_return_points_made',
    'mean_opp_return_points_made',
    'mean_second_serve_points_made',
    'mean_opp_second_serve_points_made',
    'mean_first_serve_points_made',
    'mean_opp_first_serve_points_made',
    'h2h_prior_win_percent',
    'h2h_prior_encounters',
    'prev_year_prior_encounters',
    'opp_prev_year_prior_encounters',
    'tourney_hist_prior_win_percent',
    'tourney_hist_prior_encounters',
    'opp_tourney_hist_prior_encounters',
    'mean_break_points_made',
    'mean_opp_break_points_made',
    'previous_tournament_round',
    'opp_previous_tournament_round',
    'tiebreak_win_percent',
    'opp_tiebreak_win_percent',
    'surface_experience',
    'opp_surface_experience'
]

all_attributes = list(input_attributes)
all_attributes.append('y')
all_attributes.append('year')

data, test_data = load_data(all_attributes)

# create inputs
data = (np.array(data[input_attributes]), np.array(data['y']))
test_data = (np.array(test_data[input_attributes]), np.array(test_data['y']))


X = Input((len(input_attributes),))

model = BatchNormalization()(X)
model = Dense(256, activation='tanh')(model)
model = BatchNormalization()(model)
model = Dense(256, activation='tanh')(model)
model = BatchNormalization()(model)
model = Dense(256, activation='tanh')(model)
model = Dropout(0.2)(model)
model = Dense(1, activation='sigmoid')(model)
model = Model(inputs=X, outputs=model)
model.compile(optimizer=Adam(lr=0.0001, decay=0.001), loss='mean_squared_error', metrics=['accuracy'])

model.fit(data[0], data[1], batch_size=32, epochs=30, validation_data=test_data, shuffle=True)
binary_correct, n, binary_percent, avg_error = test_model(model, test_data[0], test_data[1])

print('Correctly predicted: ' + str(binary_correct) + ' out of ' + str(n) +
      ' (' + to_percentage(binary_percent) + ')')
print('Average error: ', to_percentage(avg_error))

exit(0)

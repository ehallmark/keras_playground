from keras.layers import Dense, Reshape, Concatenate, Input, BatchNormalization, Dropout
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
    'mean_break_points_made',
    'mean_opp_break_points_made',
    'mean_break_points_saved',
    'mean_opp_break_points_saved',
    'clay',
    'grass',
    'var_first_serve_points_percent',
    'opp_var_first_serve_points_percent',
    'var_second_serve_points_percent',
    'opp_var_second_serve_points_percent',
    'var_break_points_saved_percent',
    'opp_var_break_points_saved_percent',
    'var_first_serve_return_points_percent',
    'opp_var_first_serve_return_points_percent',
    'var_second_serve_return_points_percent',
    'opp_var_second_serve_return_points_percent',
    'var_break_points_percent',
    'opp_var_break_points_percent',
    'h2h_prior_win_percent',
    'h2h_prior_encounters',
    'prev_year_prior_encounters',
    'opp_prev_year_prior_encounters',
    'tourney_hist_prior_win_percent',
    'tourney_hist_prior_encounters',
    'opp_tourney_hist_prior_encounters',
    'previous_tournament_round',
    'opp_previous_tournament_round',
    'tiebreak_win_percent',
    'opp_tiebreak_win_percent',
    'surface_experience',
    'opp_surface_experience',
    'experience',
    'opp_experience',
    'age',
    'opp_age',
    'lefty',
    'opp_lefty',
    'weight',
    'opp_weight',
    'height',
    'opp_height'
]

all_attributes = list(input_attributes)
all_attributes.append('y')
all_attributes.append('year')

data, test_data = load_data(all_attributes)

# create inputs
data = (np.array(data[input_attributes]), np.array(data['y']))
test_data = (np.array(test_data[input_attributes]), np.array(test_data['y']))

def cell(x1,x2, n_units):
    c = Concatenate()([x1,x2])
    c = BatchNormalization()(c)
    c = Dense(n_units, activation='relu')(c)
    c = Dropout(0.2)(c)
    return c

X = Input((len(input_attributes),))

hidden_units = 256
num_cells = 6

model1 = BatchNormalization()(X)
model2 = Dense(hidden_units, activation='relu')(model1)
for i in range(num_cells):
    model1 = cell(model1,model2,hidden_units)
    model2 = cell(model2,model1,hidden_units)

model = Dense(hidden_units*2, activation='tanh')(model2)
model = Dense(1, activation='sigmoid')(model)
model = Model(inputs=X, outputs=model)
model.compile(optimizer=Adam(lr=0.001, decay=0.00001), loss='mean_squared_error', metrics=['accuracy'])

model_file = 'tennis_match_keras_nn.h5'

for i in range(30):
    model.fit(data[0], data[1], batch_size=256, initial_epoch=i, epochs=i+1, validation_data=test_data, shuffle=True)
    # save
    model.save(model_file)
    print('Saved.')

binary_correct, n, binary_percent, avg_error = test_model(model, test_data[0], test_data[1])

print('Correctly predicted: ' + str(binary_correct) + ' out of ' + str(n) +
      ' (' + to_percentage(binary_percent) + ')')
print('Average error: ', to_percentage(avg_error))




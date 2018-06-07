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
meta_attributes = ['player_id', 'opponent_id', 'tournament', 'year']
for meta in meta_attributes:
    all_attributes.append(meta)


def get_all_data(test_season=2017, start_year=1996):
    all_data = load_data(all_attributes, test_season=test_season, start_year=start_year)
    data, test_data = all_data
    # create inputs
    test_meta_data = test_data[meta_attributes]
    data = (np.array(data[input_attributes]), np.array(data['y']))
    test_data = (np.array(test_data[input_attributes]), np.array(test_data['y']))
    return data, test_data, test_meta_data


if __name__ == '__main__':
    data, test_data, _ = get_all_data(test_season=2017, start_year=2000)

    def cell(x1,x2, n_units):
        c = Concatenate()([x1,x2])
        c = BatchNormalization()(c)
        c = Dense(n_units, activation='relu')(c)
        c = Dropout(0.5)(c)
        return c

    X = Input((len(input_attributes),))

    hidden_units = 512
    num_cells = 3
    batch_size = 256

    norm = BatchNormalization()(X)
    model1 = Dense(hidden_units, activation='relu')(norm)
    model2 = Dense(hidden_units, activation='relu')(norm)
    for i in range(num_cells):
        model1 = cell(model1,model2,hidden_units)
        model2 = cell(model2,model1,hidden_units)

    model = Dense(1, activation='sigmoid')(model2)
    model = Model(inputs=X, outputs=model)
    model.compile(optimizer=Adam(lr=0.001, decay=0.0001), loss='mean_squared_error', metrics=['accuracy'])

    #model_file = 'tennis_match_keras_nn.h5'
    #model_file = 'tennis_match_keras_nn_v2.h5'
    model_file = 'tennis_match_keras_nn_v3.h5'

    prev_accuracy = 0.0
    best_accuracy = 0.0
    for i in range(50):
        model.fit(data[0], data[1], batch_size=batch_size, initial_epoch=i, epochs=i+1, validation_data=test_data, shuffle=True)
        binary_correct, n, binary_percent, avg_error = test_model(model, test_data[0], test_data[1])
        print('Correctly predicted: ' + str(binary_correct) + ' out of ' + str(n) +
              ' (' + to_percentage(binary_percent) + ')')
        print('Average error: ', to_percentage(avg_error))
        if binary_percent > best_accuracy:
            best_accuracy = binary_percent
            # save
            model.save(model_file)
            print('Saved.')
        prev_accuracy = binary_percent



    print(model.summary())

    print('Most recent accuracy: ', prev_accuracy)
    print('Best accuracy: ', best_accuracy)




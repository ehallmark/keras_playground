from keras.layers import Dense, Reshape, Add, Multiply, Concatenate, Lambda, Input, BatchNormalization, Dropout
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
    'prev_h2h2_wins_opponent',
    'mean_return_points_made',
    'prev_year_avg_round',
    'tourney_hist_avg_round',
    'mean_second_serve_points_made',
    'mean_first_serve_points_made',
    'mean_break_points_made',
    'mean_break_points_saved',
    'clay',
    'grass',
    'h2h_prior_win_percent',
    'h2h_prior_encounters',
    'prev_year_prior_encounters',
    'tourney_hist_prior_win_percent',
    'tourney_hist_prior_encounters',
    'previous_tournament_round',
    'tiebreak_win_percent',
    'surface_experience',
    'experience',
    'age',
    'lefty',
    'weight',
    'height',
    'grand_slam',
    'round',
    'mean_duration',
    # Would lead to bad things like not being able to pre compute all match combinations
    'duration_prev_match',
    'elo_score'
]


opp_input_attributes = [
    'prev_h2h2_wins_opponent',
    'mean_opp_return_points_made',
    'opp_prev_year_avg_round',
    'opp_tourney_hist_avg_round',
    'mean_opp_second_serve_points_made',
    'mean_opp_first_serve_points_made',
    'mean_opp_break_points_made',
    'mean_opp_break_points_saved',
    'clay',
    'grass',
    'h2h_prior_win_percent',
    'h2h_prior_encounters',
    'opp_prev_year_prior_encounters',
    'opp_tourney_hist_prior_win_percent',
    'opp_tourney_hist_prior_encounters',
    'opp_previous_tournament_round',
    'opp_tiebreak_win_percent',
    'opp_surface_experience',
    'opp_experience',
    'opp_age',
    'opp_lefty',
    'opp_weight',
    'opp_height',
    'grand_slam',
    'round',
    'mean_opp_duration',
    # Would lead to bad things like not being able to pre compute all match combinations
    'opp_duration_prev_match',
    'opp_elo_score'
]

all_attributes = list(input_attributes)
all_attributes.append('y')
meta_attributes = ['player_id', 'opponent_id', 'tournament', 'year']
for meta in meta_attributes:
    all_attributes.append(meta)
for opp in opp_input_attributes:
    if not opp in all_attributes:
        all_attributes.append(opp)

print('Num input attrs: ', len(input_attributes))
print('Num opponent attrs: ', len(opp_input_attributes))


def get_all_data(test_season=2017, start_year=2003, tournament=None):
    all_data = load_data(all_attributes, test_season=test_season, start_year=start_year, keep_nulls=tournament is not None)
    data, test_data = all_data
    if tournament is not None:
        data = data[(data.tournament==tournament)&(data.year==test_season)]
        test_data = test_data[(test_data.tournament==tournament)&(test_data.year==test_season)]
        #test_data = test_data.sample(frac=1.0, replace=False)
        #print("TEST DATA: ", test_data)
    # create inputs
    test_meta_data = test_data[meta_attributes]
    print("Data: ", data[input_attributes])
    data = ([np.array(data[input_attributes]), np.array(data[opp_input_attributes])], np.array(data['y']))
    test_data = ([np.array(test_data[input_attributes]), np.array(test_data[opp_input_attributes])], np.array(test_data['y']))
    return data, test_data, test_meta_data


if __name__ == '__main__':
    data, test_data, _ = get_all_data(test_season=2017, start_year=1990)

    def cell(x1, norm1, x2, norm2, n_units):
        concat = Concatenate()
        batch_norm = BatchNormalization()
        dense = Dense(n_units, activation='tanh')
        dropout = Dropout(0.5)
        norm1 = concat([x1, norm1])
        norm1 = batch_norm(norm1)
        norm1 = dense(norm1)
        norm1 = dropout(norm1)
        norm2 = concat([x2, norm2])
        norm2 = batch_norm(norm2)
        norm2 = dense(norm2)
        norm2 = dropout(norm2)
        return norm1, x1, norm2, x2

    X1 = Input((len(input_attributes),))
    X2 = Input((len(opp_input_attributes),))

    hidden_units = 128 #len(input_attributes)*2
    num_cells = 6
    batch_size = 256

    norm = BatchNormalization()
    norm1 = norm(X1)
    norm2 = norm(X2)
    model = Dense(hidden_units, activation='tanh')
    model1 = model(norm1)
    model2 = model(norm2)
    for i in range(num_cells):
        model1, norm1, model2, norm2 = cell(model1, norm1, model2, norm2, hidden_units)

    model = Dense(hidden_units, activation='tanh')
    model1 = model(model1)
    model2 = model(model2)
    model = Add()([model2, Lambda(lambda x: -x)(model1)])
    model = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=[X1, X2], outputs=model)
    model.compile(optimizer=Adam(lr=0.001, decay=0.01), loss='mean_squared_error', metrics=['accuracy'])
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


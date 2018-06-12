from keras.layers import Dense, Reshape, Add, Multiply, Concatenate, Lambda, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras as k
from models.atp_tennis.TennisMatchOutcomeLogit import load_data, to_percentage
import numpy as np
np.random.seed(23952)

def test_model(model, x, y):
    predictions = model.predict(x)
    avg_error = 0
    avg_error += np.mean(np.abs(np.array(y[0]) - np.array(predictions[0])))
    avg_error += np.mean(np.abs(np.array(y[1]) - np.array(predictions[1])))
    avg_error /= 2
    return avg_error


input_attributes = [
    'ml_odds1',
    'prev_h2h2_wins_opponent',
    'mean_return_points_made',
    'prev_year_avg_round',
    'tourney_hist_avg_round',
    'mean_second_serve_points_made',
    'mean_first_serve_points_made',
    'mean_break_points_made',
    'mean_break_points_saved',
    #'clay',
    #'grass',
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
    #'mean_duration',
    # Would lead to bad things like not being able to pre compute all match combinations
    #'duration_prev_match',
    'elo_score'
]


opp_input_attributes = [
    'ml_odds2',
    'prev_h2h2_wins_opponent',
    'mean_opp_return_points_made',
    'opp_prev_year_avg_round',
    'opp_tourney_hist_avg_round',
    'mean_opp_second_serve_points_made',
    'mean_opp_first_serve_points_made',
    'mean_opp_break_points_made',
    'mean_opp_break_points_saved',
    #'clay',
    #'grass',
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
    #'mean_opp_duration',
    # Would lead to bad things like not being able to pre compute all match combinations
    #'opp_duration_prev_match',
    'opp_elo_score'
]

all_attributes = list(input_attributes)
all_attributes.append('ml_return1')
all_attributes.append('ml_return2')
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
    data = ([np.array(data[input_attributes]), np.array(data[opp_input_attributes])], [np.array(data['ml_return1']), np.array(data['ml_return2'])])
    test_data = ([np.array(test_data[input_attributes]), np.array(test_data[opp_input_attributes])], [np.array(test_data['ml_return1']), np.array(test_data['ml_return2'])])
    return data, test_data, test_meta_data


if __name__ == '__main__':
    data, test_data, _ = get_all_data(test_season=2016, start_year=2003)
    X1 = Input((len(input_attributes),))
    X2 = Input((len(opp_input_attributes),))

    hidden_units = 256  # len(input_attributes)*2
    num_cells = 3
    batch_size = 128
    dropout = 0.5
    load_previous = False
    if load_previous:
        model = k.models.load_model('tennis_match_keras_nn_v4.h5')
        model.compile(optimizer=Adam(lr=0.001, decay=0.0001), loss='mean_squared_error', metrics=['accuracy'])
    else:
        def cell(x1, x2, n_units, dropout=0.5):
            concat = Concatenate()
            #batch_norm = BatchNormalization()
            dense = Dense(n_units, activation='tanh')
            dropout_layer = Dropout(dropout)
            #norm1 = concat([x1, norm1])
            #norm1 = batch_norm(x1)
            norm1 = dense(x1)
            norm1 = dropout_layer(norm1)
            #norm2 = concat([x2, norm2])
            #norm2 = batch_norm(x2)
            norm2 = dense(x2)
            norm2 = dropout_layer(norm2)
            return norm1, norm2

        norm = BatchNormalization()
        model1 = norm(X1)
        model2 = norm(X2)

        model = Dense(hidden_units, activation='tanh')
        model1 = Dropout(dropout)(model(model1))
        model2 = Dropout(dropout)(model(model2))
        out1 = Add()([model1, Lambda(lambda x: -x)(model2)])
        out2 = Add()([model2, Lambda(lambda x: -x)(model1)])
        model = BatchNormalization()
        out1 = model(out1)
        out2 = model(out2)
        model = Dense(hidden_units, activation='tanh')
        out1 = model(out1)
        out2 = model(out2)
        out_pre = Dense(10, activation='tanh')
        out = Dense(1, activation='linear')
        out1 = out(out_pre(out1))
        out2 = out(out_pre(out2))
        model = Model(inputs=[X1, X2], outputs=[out1, out2])
        model.compile(optimizer=Adam(lr=0.0001, decay=0.0001), loss='mean_squared_error', metrics=['accuracy'])

    model_file = 'tennis_match_keras_end2end_v1.h5'
    prev_error = None
    best_error = None
    for i in range(50):
        model.fit(data[0], data[1], batch_size=batch_size, initial_epoch=i, epochs=i+1, validation_data=test_data, shuffle=True)
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


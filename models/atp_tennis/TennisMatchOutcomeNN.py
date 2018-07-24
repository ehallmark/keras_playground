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
    avg_error += np.mean(np.abs(np.array(y).flatten() - np.array(predictions).flatten()))
    return avg_error


# previous year quality
input_attributes0 = [
    'h2h_prior_win_percent',
    'tourney_hist_avg_round',
    'prev_year_avg_round',
    'prev_year_prior_losses',
    'prior_year_match_closeness',

    # prior quarter
    'prior_quarter_games_per_set',
    'prior_quarter_victories',

    # player qualities
    'elo_score_weighted',
    #'age',
    'surface_experience',
    #'height',
    'best_year',

    # match stats
    'tiebreak_win_percent',
    'major_encounters',
    'master_encounters',
    'encounters_250',
    'challenger_encounters',
    'mean_faults',
    'mean_aces',
    'mean_service_points_won',
    'mean_return_points_made',

    # previous match
    'previous_games_total2',
    'had_qualifier',
    'wild_card',
    'seeded',
]

# opponent attrs
opp_input_attributes0 = ['opp_'+attr for attr in input_attributes0]
opp_input_attributes0.remove('opp_h2h_prior_win_percent')
input_attributes = input_attributes0 + opp_input_attributes0


additional_attributes = [
    'clay',
    'grass',
    'tournament_rank',
    'first_round',
]

for attr in additional_attributes:
    input_attributes.append(attr)

all_attributes = list(input_attributes)
all_attributes.append('y')
all_attributes.append('spread')
meta_attributes = ['player_id', 'opponent_id', 'tournament', 'year', 'clay', 'tournament_rank']
for meta in meta_attributes:
    if meta not in all_attributes:
        all_attributes.append(meta)


if __name__ == '__main__':
    data = load_data(all_attributes, test_season='2012-01-01', start_year='1995-01-01')
    test_data = load_data(all_attributes, test_season='2012-12-31', start_year='2012-01-01')
    # data = data[data.clay<0.5]
    # test_data = test_data[test_data.clay<0.5]
    X1 = Input((len(input_attributes),))

    data = (np.array(data[input_attributes]), np.array(data['y']))
    test_data = (np.array(test_data[input_attributes]), np.array(test_data['y']))

    hidden_units = 1000  # len(input_attributes)*2
    num_cells = 3
    batch_size = 128
    dropout = 0.5
    load_previous = False
    if load_previous:
        model = k.models.load_model('tennis_match_keras_nn_v5.h5')
        model.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        def cell(x1, n_units, dropout=0.5):
            batch_norm = BatchNormalization()
            dense = Dense(n_units, activation='tanh')
            dropout_layer = Dropout(dropout)
            norm1 = dense(x1)
            norm1 = batch_norm(norm1)
            norm1 = dropout_layer(norm1)
            return norm1

        norm = BatchNormalization()
        model1 = norm(X1)
        for i in range(num_cells):
            model1 = cell(model1, hidden_units)

        out1 = Dense(1, activation='sigmoid')(model1)
        model = Model(inputs=X1, outputs=out1)
        model.compile(optimizer=Adam(lr=0.001, decay=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model_file = 'tennis_match_keras_nn_v5.h6'
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


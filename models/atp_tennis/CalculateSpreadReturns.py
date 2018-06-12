import keras as k
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import models.atp_tennis.TennisMatchOutcomeNN as tennis_model
from models.atp_tennis.TennisMatchOutcomeNN import test_model,to_percentage
from models.simulation.Simulate import simulate_spread


def load_predictions_and_actuals(model, test_year=2018):
    all_data = tennis_model.get_all_data(test_year)
    test_meta_data = all_data[2]
    test_data = all_data[1]
    test_labels = test_data[1]
    avg_error = test_model(model, test_data[0], test_labels)
    print('Average error: ', to_percentage(avg_error))
    predictions = model.predict(test_data[0])
    predictions[0] = predictions[0].flatten()
    predictions[1] = predictions[1].flatten()
    return predictions, test_labels, test_meta_data


def load_betting_data(betting_sites, test_year=2018):
    conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
    betting_data = pd.read_sql('''
        select year,tournament,team1,team2,
        price1,
        price2,
        spread1,
        spread2
        from atp_tennis_betting_link_spread 
        where year={{YEAR}} and book_name in ({{BOOK_NAMES}})
    '''.replace('{{YEAR}}', str(test_year)).replace('{{BOOK_NAMES}}', '\''+'\',\''.join(betting_sites)+'\''), conn)
    return betting_data


def betting_decision(victory_prediction, spread_prediction, odds, spread, underdog, parameters={}):
    if underdog:
        if victory_prediction > odds + parameters['betting_epsilon1'] and spread - spread_prediction < parameters['spread_epsilon']:
            # check spread and prediction
            return True

        return False
    else:
        if victory_prediction > odds + parameters['betting_epsilon2'] and spread + spread_prediction > parameters['spread_epsilon']:
            return True
        return False


def parameter_update_func(parameters):
    parameters['max_loss_percent'] = 0.05
    parameters['betting_epsilon1'] = float(0.15 + (np.random.rand(1) * 0.1 - 0.05))
    parameters['betting_epsilon2'] = float(0.25 + (np.random.rand(1) * 0.1 - 0.05))
    parameters['spread_epsilon'] = float(1.5 + (np.random.rand(1) * 8.0))
    parameters['max_price_plus'] = 200
    parameters['max_price_minus'] = -180


def predictor_func(i):
    return predictions[0][i]


def predict_spread_func(i):
    return predictions[1][i]


def actual_label_func(i):
    return test_labels[0][i]


def actual_spread_func(i):
    return test_labels[1][i]


if __name__ == '__main__':
    # define variables
    test_year = 2018
    price_str = 'price'
    parameters = {}
    np.random.seed(1)
    num_trials = 50
    betting_sites = [
        'Bovada',
        '5Dimes',
        'BetOnline'
    ]
    betting_data = load_betting_data(betting_sites, test_year=test_year)
    model = k.models.load_model('tennis_match_keras_nn_v4.h5')
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
    print(model.summary())
    predictions, test_labels, test_meta_data = load_predictions_and_actuals(model, test_year=test_year)

    # run simulation
    simulate_spread(predictor_func, predict_spread_func, actual_label_func, actual_spread_func,
                    parameter_update_func, betting_decision, test_meta_data, betting_data, parameters,
                    price_str, num_trials)

    exit(0)


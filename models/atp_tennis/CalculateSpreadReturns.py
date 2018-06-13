import keras as k
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import models.atp_tennis.TennisMatchOutcomeNN as tennis_model
from models.atp_tennis.TennisMatchOutcomeNN import test_model,to_percentage
from models.simulation.Simulate import simulate_spread
from models.genetics.GeneticAlgorithm import GeneticAlgorithm, Solution


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
        book_name,
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
    parameters['max_loss_percent'] = float(0.01 + np.random.rand(1) * 0.1)
    parameters['betting_epsilon1'] = float(0.25 + (np.random.rand(1) * 0.2 - 0.1))
    parameters['betting_epsilon2'] = float(0.15 + (np.random.rand(1) * 0.2 - 0.1))
    parameters['spread_epsilon'] = float(2.0 + (np.random.rand(1) * 1.))
    parameters['max_price_plus'] = float(0.0 + np.random.rand(1) * 400.0)
    parameters['max_price_minus'] = float(0.0 - np.random.rand(1) * 400.0)


def predictor_func(i):
    return predictions[0][i]


def predict_spread_func(i):
    return predictions[1][i]


def actual_label_func(i):
    return test_labels[0][i]


def actual_spread_func(i):
    return test_labels[1][i]


if __name__ == '__main__':

    class SpreadSolution(Solution):
        def __init__(self, parameters):
            self.parameters = parameters.copy()

        def betting_epsilon_func(self, price):
            if price > 0:
                return self.parameters['betting_epsilon1']
            else:
                return self.parameters['betting_epsilon2']

        def score(self, data):
            # run simulation
            return simulate_spread(predictor_func, predict_spread_func, actual_label_func, actual_spread_func,
                                   parameter_update_func, betting_sites, betting_decision, data[0], data[1], self.parameters,
                                   price_str, 1)

        def mutate(self):
            mutated_parameters = self.parameters.copy()
            second_copy = self.parameters.copy()
            parameter_update_func(second_copy)
            for k, v in second_copy.items():
                if np.random.rand(1) < 2.0/float(len(second_copy)):
                    mutated_parameters[k] = (mutated_parameters[k]+v)/2.
            return SpreadSolution(mutated_parameters)

        def cross_over(self, other):
            cross_parameters = self.parameters.copy()
            for k in cross_parameters:
                if np.random.rand(1) < 0.5:
                    cross_parameters[k] = other.parameters[k]
            return SpreadSolution(cross_parameters)


    def solution_generator():
        parameters = {}
        parameter_update_func(parameters)
        return SpreadSolution(parameters)

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
    model = k.models.load_model('tennis_match_keras_nn_v5.h5')
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
    print(model.summary())
    predictions, test_labels, test_meta_data = load_predictions_and_actuals(model, test_year=test_year)

    # run simulation
    genetic_algorithm = GeneticAlgorithm(solution_generator=solution_generator)
    data = (test_meta_data, betting_data)
    genetic_algorithm.fit(data)
    exit(0)


import keras as k
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import models.atp_tennis.TennisMatchOutcomeNN as tennis_model
from models.atp_tennis.TennisMatchOutcomeNN import test_model,to_percentage
from models.simulation.Simulate import simulate_spread
from models.genetics.GeneticAlgorithm import GeneticAlgorithm, Solution
from models.atp_tennis.CalculateMoneyLineReturns import Return

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


def parameter_tweak_function(parameters):
    parameters['max_loss_percent'] += float(0.01 - np.random.rand(1) * 0.02)
    parameters['betting_epsilon1'] += float(0.02 - np.random.rand(1) * 0.04)
    parameters['betting_epsilon2'] += float(0.02 - np.random.rand(1) * 0.04)
    parameters['spread_epsilon'] += float(0.5 - (np.random.rand(1) * 1.))
    parameters['max_price_plus'] += float(50.0 - np.random.rand(1) * 100.0)
    parameters['max_price_minus'] += float(50.0 - np.random.rand(1) * 100.0)


def predictor_func(i):
    return predictions[0][i]


def predict_spread_func(i):
    return predictions[1][i]


def actual_label_func(i):
    return labels[0][i]


def actual_spread_func(i):
    return labels[1][i]


def predictor_func_test(i):
    return predictions_test[0][i]


def predict_spread_func_test(i):
    return predictions_test[1][i]


def actual_label_func_test(i):
    return labels_test[0][i]


def actual_spread_func_test(i):
    return labels_test[1][i]


if __name__ == '__main__':

    class SpreadSolution(Solution):
        def __init__(self, parameters, return_class):
            self.parameters = parameters.copy()
            self.return_class = return_class

        def betting_epsilon_func(self, price):
            if price > 0:
                return self.parameters['betting_epsilon1']
            else:
                return self.parameters['betting_epsilon2']

        def score(self, data):
            # run simulation
            test_score = simulate_spread(predictor_func_test, predict_spread_func_test, actual_label_func_test,
                                         actual_spread_func_test, parameter_tweak_function, betting_decision,
                                         data[1][0], data[1][1], self.parameters, price_str, num_samples_per_solution)
            self.return_class.add_return(test_score)
            print('Avg test score: ', self.return_class.get_avg())
            return simulate_spread(predictor_func, predict_spread_func, actual_label_func, actual_spread_func,
                                   parameter_tweak_function, betting_decision, data[0][0], data[0][1], self.parameters,
                                   price_str, num_samples_per_solution)

        def mutate(self):
            mutated_parameters = self.parameters.copy()
            second_copy = self.parameters.copy()
            parameter_update_func(second_copy)
            for k, v in second_copy.items():
                if np.random.rand(1) < 2.0/float(len(second_copy)):
                    mutated_parameters[k] = (mutated_parameters[k]+v)/2.
            return SpreadSolution(mutated_parameters, self.return_class)

        def cross_over(self, other):
            cross_parameters = self.parameters.copy()
            for k in cross_parameters:
                if np.random.rand(1) < 0.5:
                    cross_parameters[k] = other.parameters[k]
            return SpreadSolution(cross_parameters, self.return_class)


    def solution_generator():
        parameters = {}
        parameter_update_func(parameters)
        return SpreadSolution(parameters, returns)


    def load_data(test_year):
        betting_sites = [
            'Bovada',
            '5Dimes',
            'BetOnline'
        ]
        betting_data = load_betting_data(betting_sites, test_year=test_year)
        predictions, test_labels, test_meta_data = load_predictions_and_actuals(model, test_year=test_year)
        test_data = []
        for betting_site in betting_sites:
            original_len = test_meta_data.shape[0]
            df = pd.DataFrame.merge(
                test_meta_data,
                betting_data[betting_data.book_name == betting_site],
                'left',
                left_on=['year', 'player_id', 'opponent_id', 'tournament'],
                right_on=['year', 'team1', 'team2', 'tournament'])
            after_len = df.shape[0]
            test_data.append(df)
            if original_len != after_len:
                print('Join data has different length... ', original_len, after_len)
                exit(1)
        return (test_meta_data, test_data), predictions, test_labels


    # define variables
    test_year = 2018
    train_year = 2017
    price_str = 'price'
    parameters = {}
    np.random.seed(1)
    num_epochs = 50
    num_samples_per_solution = 2
    model = k.models.load_model('tennis_match_keras_nn_v5.h5')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    print(model.summary())
    returns = Return()

    # run simulation
    genetic_algorithm = GeneticAlgorithm(solution_generator=solution_generator)
    data, predictions, labels = load_data(train_year)
    test_data, predictions_test, labels_test = load_data(test_year)
    genetic_algorithm.fit((data, test_data), num_solutions=10, num_epochs=num_epochs)
    exit(0)


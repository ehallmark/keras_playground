import keras as k
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from models.simulation.Simulate import simulate_money_line
import models.atp_tennis.TennisMatchOutcomeNN as tennis_model
from models.atp_tennis.TennisMatchOutcomeNN import test_model,to_percentage
from models.genetics.GeneticAlgorithm import GeneticAlgorithm, Solution


class Return:
    def __init__(self):
        self.total_return = 0.0
        self.count = 0

    def add_return(self, ret):
        self.count += 1
        self.total_return += ret

    def get_avg(self):
        return self.total_return / self.count


if __name__ == '__main__':
    def load_predictions_and_actuals(model, test_year=2018):
        all_data = tennis_model.get_all_data(test_year)
        test_meta_data = all_data[2]
        test_data = all_data[1]
        test_labels = test_data[1]
        avg_error = test_model(model, test_data[0], test_labels)
        print('Average error: ', to_percentage(avg_error))
        print('Test Meta Data Size: ', test_meta_data.shape[0])
        predictions = model.predict(test_data[0])
        predictions[0] = predictions[0].flatten()
        predictions[1] = predictions[1].flatten()
        return predictions, test_labels, test_meta_data


    def load_betting_data(betting_sites, test_year=2018):
        conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
        betting_data = pd.read_sql('''
            select year,tournament,team1,team2,
            min(price1) as min_price1, 
            max(price1) as max_price1,
            min(price2) as min_price2,
            max(price2) as max_price2,
            sum(price1)/count(price1) as avg_price1,
            sum(price2)/count(price2) as avg_price2
            from atp_tennis_betting_link 
            where year={{YEAR}} and book_name in ({{BOOK_NAMES}})
            group by year,tournament,team1,team2
        '''.replace('{{YEAR}}', str(test_year)).replace('{{BOOK_NAMES}}', '\''+'\',\''.join(betting_sites)+'\''), conn)
        return betting_data


    def parameter_update_func(parameters):
        parameters['max_loss_percent'] = float(0.01 + np.random.rand(1) * 0.1)
        parameters['betting_epsilon1'] = float(0.15 + (np.random.rand(1) * 0.2 - 0.1))
        parameters['betting_epsilon2'] = float(0.20 + (np.random.rand(1) * 0.2 - 0.1))
        parameters['max_price_plus'] = float(0.0 + np.random.rand(1) * 800.)
        parameters['max_price_minus'] = float(0.0 - np.random.rand(1) * 500.0)


    def parameter_tweak_func(parameters):
        parameters['max_loss_percent'] += float(0.01 - np.random.rand(1) * 0.02)
        parameters['betting_epsilon1'] += float(0.02 - np.random.rand(1) * 0.04)
        parameters['betting_epsilon2'] += float(0.02 - np.random.rand(1) * 0.04)
        parameters['max_price_plus'] += float(50.0 - np.random.rand(1) * 100.)
        parameters['max_price_minus'] += float(50.0 - np.random.rand(1) * 100.0)


    def predictor_func(i):
        return predictions[0][i]


    def actual_label_func(i):
        return labels[0][i]


    def predictor_func_test(i):
        return predictions_test[0][i]


    def actual_label_func_test(i):
        return labels_test[0][i]


    class MoneyLineSolution(Solution):
        def __init__(self, parameters, return_class):
            self.return_class = return_class
            self.parameters = parameters.copy()

        def betting_epsilon_func(self, price):
            if price > 0:
                return self.parameters['betting_epsilon1']
            else:
                return self.parameters['betting_epsilon2']

        def score(self, data):
            # run simulation
            test_score = simulate_money_line(predictor_func_test, actual_label_func_test, parameter_tweak_func,
                                       self.betting_epsilon_func, data[1], self.parameters,
                                       price_str, 2)
            self.return_class.add_return(test_score)
            print('Avg test score: ', self.return_class.get_avg())
            return simulate_money_line(predictor_func, actual_label_func, parameter_tweak_func,
                                       self.betting_epsilon_func, data[0], self.parameters,
                                       price_str, num_samples_per_solution)

        def mutate(self):
            mutated_parameters = self.parameters.copy()
            second_copy = self.parameters.copy()
            parameter_update_func(second_copy)
            for k, v in second_copy.items():
                if np.random.rand(1) < 2.0/float(len(second_copy)):
                    mutated_parameters[k] = (mutated_parameters[k]+v)/2.
            return MoneyLineSolution(mutated_parameters, self.return_class)

        def cross_over(self, other):
            cross_parameters = self.parameters.copy()
            for k in cross_parameters:
                if np.random.rand(1) < 0.5:
                    cross_parameters[k] = other.parameters[k]
            return MoneyLineSolution(cross_parameters, self.return_class)


    def solution_generator():
        parameters = {}
        parameter_update_func(parameters)
        return MoneyLineSolution(parameters, returns)


    def load_data(test_year):
        predictions, test_labels, test_meta_data = load_predictions_and_actuals(model, test_year=test_year)
        betting_sites = ['Bovada', '5Dimes', 'BetOnline']
        betting_data = load_betting_data(betting_sites, test_year=test_year)
        original_len = test_meta_data.shape[0]
        test_meta_data = pd.DataFrame.merge(
            test_meta_data,
            betting_data,
            'left',
            left_on=['year', 'player_id', 'opponent_id', 'tournament'],
            right_on=['year', 'team1', 'team2', 'tournament'])
        after_len = test_meta_data.shape[0]
        if original_len != after_len:
            print('Join data has different length... ', original_len, after_len)
            exit(1)
        return predictions, test_labels, test_meta_data

    # define vars
    train_year = 2017
    test_year = 2018
    price_str = 'max_price'
    num_epochs = 50
    num_samples_per_solution = 3
    parameters = {}
    model = k.models.load_model('tennis_match_keras_nn_v5.h5')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    predictions, labels, meta_data = load_data(test_year=train_year)
    predictions_test, labels_test, meta_data_test = load_data(test_year=test_year)

    returns = Return()
    genetic_algorithm = GeneticAlgorithm(solution_generator=solution_generator)

    # fit data
    genetic_algorithm.fit((meta_data, meta_data_test), num_epochs=num_epochs, num_solutions=10)
    exit(0)



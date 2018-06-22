from models.atp_tennis.TennisMatchOutcomeLogit import test_model, to_percentage
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from models.genetics.GeneticAlgorithm import Solution, GeneticAlgorithm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVR
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
from models.atp_tennis.TennisMatchOutcomeSklearnModels import load_outcome_model, load_spread_model
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from models.atp_tennis.TennisMatchOutcomeLogit import input_attributes as outcome_input_attributes
from models.atp_tennis.TennisMatchOutcomeLogit import input_attributes_spread as spread_input_attributes
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from models.simulation.Simulate import simulate_money_line


betting_input_attributes = list(outcome_input_attributes)

y_str = 'y'
all_attributes = list(betting_input_attributes)
all_attributes.append(y_str)
meta_attributes = ['player_id', 'opponent_id', 'tournament', 'year']
for meta in meta_attributes:
    all_attributes.append(meta)


def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        prob_pos = model.predict_proba(X)[:, 1]
    else:  # use decision function
        prob_pos = model.decision_function(X)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    return prob_pos


def load_outcome_predictions_and_actuals(attributes, model=None, spread_model=None, test_year=2018, num_test_years=3, start_year=2005):
    data, _ = tennis_model.get_all_data(attributes, test_season=test_year-num_test_years+1, start_year=start_year)
    test_data, _ = tennis_model.get_all_data(attributes, test_season=test_year+1, start_year=test_year+1-num_test_years)
    X = np.array(data[outcome_input_attributes].iloc[:, :])
    X_test = np.array(test_data[outcome_input_attributes].iloc[:, :])
    X_spread = np.array(data[spread_input_attributes].iloc[:, :])
    X_test_spread = np.array(test_data[spread_input_attributes].iloc[:, :])
    if model is not None:
        data = data.assign(predictions=pd.Series(predict_proba(model, X)).values)
        test_data = test_data.assign(predictions=pd.Series(predict_proba(model, X_test)).values)
    if spread_model is not None:
        data = data.assign(spread_predictions=pd.Series(spread_model.predict(X_spread)).values)
        test_data = test_data.assign(spread_predictions=pd.Series(spread_model.predict(X_test_spread)).values)
    return data, test_data


def load_betting_data(betting_sites, test_year=2018):
    conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
    betting_data = pd.read_sql('''
         select year,tournament,team1,team2,
         min(price1) as min_price1, 
         max(price1) as max_price1,
         min(price2) as min_price2,
         max(price2) as max_price2,
         sum(odds1)/count(*) as odds1,
         sum(odds2)/count(*) as odds2,
         mode() within group(order by betting_date) as betting_date
         from atp_tennis_betting_link 
         where year<={{YEAR}} and book_name in ({{BOOK_NAMES}})
         group by year,tournament,team1,team2
     '''.replace('{{YEAR}}', str(test_year)).replace('{{BOOK_NAMES}}', '\'' + '\',\''.join(betting_sites) + '\''), conn)
    return betting_data


def load_data(start_year, test_year, num_test_years):
    attributes = list(tennis_model.all_attributes)
    data, test_data = load_outcome_predictions_and_actuals(attributes, test_year=test_year, num_test_years=num_test_years,
                                                           start_year=start_year)
    # merge betting data in memory
    betting_sites = ['Bovada',
                     '5Dimes',  # 5Dimes available for money line usually (but not for spreads)
                     'BetOnline']
    betting_data = load_betting_data(betting_sites, test_year=test_year)
    test_data = pd.DataFrame.merge(
        test_data,
        betting_data,
        'left',
        left_on=['year', 'player_id', 'opponent_id', 'tournament'],
        right_on=['year', 'team1', 'team2', 'tournament'],
        validate='1:1'
    )
    data = pd.DataFrame.merge(
        data,
        betting_data,
        'left',
        left_on=['year', 'player_id', 'opponent_id', 'tournament'],
        right_on=['year', 'team1', 'team2', 'tournament'],
        validate='1:1'
    )
    #data.sort_values(by=['betting_date'], inplace=True, ascending=True, kind='mergesort')
    #test_data.sort_values(by=['betting_date'], inplace=True, ascending=True, kind='mergesort')
    #print('Sorted test data.......', test_data)
    return data, test_data


def bet_func(epsilon, parameters):
    def bet_func_helper(price, odds, prediction):
        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)
        if odds < parameters['min_odds']:
            return 0
        if odds > parameters['max_odds']:
            return 0
        if price > 0:
            expectation_implied = odds * price + (1. - odds) * -100.
            expectation = prediction * price + (1. - prediction) * -100.
            expectation /= 100.
            expectation_implied /= 100.
        else:
            expectation_implied = odds * 100. + (1. - odds) * price
            expectation = prediction * 100. + (1. - prediction) * price
            expectation /= -price
            expectation_implied /= -price
        # print('Expectation:', expectation, ' Implied: ', expectation_implied)
        if expectation > epsilon:
            return 1. + expectation
        else:
            return 0

    return bet_func_helper


def new_random_parameters():
    model_parameters = {}
    model_parameters['model_to_epsilon'] = {
        'Logit Regression': float(np.random.rand(1)),
        'Naive Bayes': float(np.random.rand(1)),
        'Random Forest': float(np.random.rand(1)),
        'Average': float(np.random.rand(1)),
        'QDA': float(np.random.rand(1)),
        'Support Vector': float(np.random.rand(1)),
        'K Neighbors': float(np.random.rand(1)),
    }
    model_parameters['model_weights'] = {
        'Logit Regression': float(np.random.rand(1)),
        'Naive Bayes': float(np.random.rand(1)),
        'Random Forest': float(np.random.rand(1)),
        'Average': float(np.random.rand(1)),
        'QDA': float(np.random.rand(1)),
        'Support Vector': float(np.random.rand(1)),
        'K Neighbors': float(np.random.rand(1)),
    }
    model_parameters['min_odds'] = float(np.random.rand(1))*0.5
    model_parameters['max_odds'] = float(np.random.rand(1))*0.5+0.5
    return model_parameters


class MoneyLineSolution(Solution):
    def __init__(self, parameters):
        self.parameters = parameters.copy()

    def score(self, data):
        # run simulation
        return test(self.parameters, data[0], data[1])

    def mutate(self):
        mutated_parameters = self.parameters.copy()
        second_copy = new_random_parameters()
        mutated_weights = mutated_parameters['model_weights']
        mutated_epsilons = mutated_parameters['model_to_epsilon']
        second_weights = second_copy['model_weights']
        second_epsilons = second_copy['model_to_epsilon']
        for k, v in second_weights.items():
            if np.random.rand(1) < 2.0/float(len(second_weights)):
                mutated_weights[k] = (mutated_weights[k]+v)/2.
        for k, v in second_epsilons.items():
            if np.random.rand(1) < 2.0/float(len(second_epsilons)):
                mutated_epsilons[k] = (mutated_epsilons[k]+v)/2.
        if np.random.rand(1) < 0.25:
            mutated_parameters['max_odds'] = (mutated_parameters['max_odds'] + second_copy['max_odds'])/2.
        if np.random.rand(1) < 0.25:
            mutated_parameters['min_odds'] = (mutated_parameters['min_odds'] + second_copy['min_odds'])/2.
        return MoneyLineSolution(mutated_parameters)

    def cross_over(self, other):
        cross_parameters = self.parameters.copy()
        for k in cross_parameters:
            if np.random.rand(1) < 0.5:
                cross_parameters[k] = other.parameters[k]

        return MoneyLineSolution(cross_parameters)


def solution_generator():
    parameters = new_random_parameters()
    return MoneyLineSolution(parameters)


def test(model_parameters, data, test_data):
    num_tests = 0
    returns = 0.0
    #for num_test_years in [1]:
    #    for test_year in [2016, 2017, 2018]:
    num_tests += 1
    price_str = 'max_price'
    num_tests = 1
    graph = False
    lr = lambda: LogisticRegression()
    svm = lambda: LinearSVC()
    rf = lambda: RandomForestClassifier(n_estimators=300, max_depth=10)
    nb = lambda: GaussianNB()
    qda = lambda: QuadraticDiscriminantAnalysis()
    kn = lambda: KNeighborsClassifier(n_neighbors=10)
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    X_train = np.array(data[betting_input_attributes].iloc[:, :])
    y_train = np.array(data[y_str].iloc[:]).flatten()
    X_test = np.array(test_data[betting_input_attributes].iloc[:, :])
    y_test = np.array(test_data[y_str].iloc[:]).flatten()
    for _model, name in [
                    #(lr, 'Logit Regression'),
                    #(svm, 'Support Vector'),
                    (nb, 'Naive Bayes'),
                    #(rf, 'Random Forest'),
                    #(qda, 'QDA'),
                    #(kn, 'K Neighbors')
                ]:
        all_predictions = []
        weight = model_parameters['model_weights'][name]
        total_weight = 0.0
        print('with Betting Model: ', name)
        #print("Shapes: ", X_train.shape, X_test.shape)
        for i in range(30):
            total_weight += weight
            model = _model()
            np.random.seed(i)
            np.random.shuffle(X_train)
            np.random.seed(i)
            np.random.shuffle(y_train)
            X_train_sample = X_train[0:round(X_train.shape[0]/2)]
            y_train_sample = y_train[0:round(y_train.shape[0]/2)]
            model.fit(X_train_sample, y_train_sample)
            #binary_correct, n, binary_percent, avg_error = test_model(model, X_test, y_test)
            #print('Correctly predicted: ' + str(binary_correct) + ' out of ' + str(n) +
            #      ' (' + to_percentage(binary_percent) + ')')
            prob_pos = predict_proba(model, X_test)
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_test, prob_pos, n_bins=10)
            all_predictions.append(prob_pos*weight)
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                     label="%s" % (name,))

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                     histtype="step", lw=2)

            # actual returns on test data
            #test_return, num_bets = simulate_money_line(lambda j: prob_pos[j], lambda j: test_data.iloc[j]['y'],
            #                                  bet_func(model_parameters['model_to_epsilon'][name], model_parameters), test_data,
            #                                  price_str, num_tests, sampling=0, shuffle=True)
            #print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:', to_percentage(avg_error), ' Test years:', num_test_years, ' Test year:', test_year)

        avg_predictions = np.vstack(all_predictions).sum(0) / total_weight
        _, _, _, avg_error = tennis_model.score_predictions(avg_predictions, test_data[y_str])
        test_return, num_bets = simulate_money_line(lambda j: avg_predictions[j], lambda j: test_data[y_str].iloc[j],
                                                    bet_func(model_parameters['model_to_epsilon']['Average'],
                                                             model_parameters), test_data,
                                                    price_str, num_tests, sampling=0, shuffle=True, verbose=False)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        if graph:
            plt.tight_layout()
            plt.show()

        print('Avg model: ', model_parameters)
        print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:', to_percentage(avg_error),
              )# ' Test years:', num_test_years, ' Test year:', test_year)
        print('---------------------------------------------------------')
        returns += test_return
        if num_bets < 100:
            return -100000.
        return returns/num_tests


if __name__ == '__main__':
    start_year = 1996
    num_test_years = 1
    test_year = 2018
    num_epochs = 10
    data, test_data = load_data(start_year=start_year, num_test_years=num_test_years, test_year=test_year)
    genetic_algorithm = GeneticAlgorithm(solution_generator=solution_generator)
    genetic_algorithm.fit((data, test_data), num_solutions=10, num_epochs=num_epochs)

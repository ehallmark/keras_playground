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


betting_input_attributes = [
        'prev_h2h2_wins_player',
        'prev_h2h2_wins_opponent',
        #'mean_duration',
        #'mean_opp_duration',
        'mean_return_points_made',
        'mean_opp_return_points_made',
        'mean_second_serve_points_made',
        'mean_opp_second_serve_points_made',
        'h2h_prior_win_percent',
        'prev_year_prior_encounters',
        'opp_prev_year_prior_encounters',
        #'prev_year_avg_round',
        #'opp_prev_year_avg_round',
        #'opp_tourney_hist_avg_round',
        #'tourney_hist_avg_round',
        'tourney_hist_prior_encounters',
        'opp_tourney_hist_prior_encounters',
        'mean_break_points_made',
        'mean_opp_break_points_made',
        #'previous_tournament_round',
        #'opp_previous_tournament_round',
        'tiebreak_win_percent',
        'opp_tiebreak_win_percent',
        'surface_experience',
        'opp_surface_experience',
        'experience',
        'opp_experience',
        'age',
        'opp_age',
        #'lefty',
        #'opp_lefty',
        #'weight',
        #'opp_weight',
        'height',
        'opp_height',
        #'duration_prev_match',
        #'opp_duration_prev_match',
        'elo_score',
        'opp_elo_score'
    ]

betting_only_attributes = [
    'odds1',
    'odds2',
    'predictions',
    #'spread_predictions'
]

for attr in betting_only_attributes:
    betting_input_attributes.append(attr)

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


def load_outcome_predictions_and_actuals(attributes, test_tournament=None, model=None, spread_model=None, test_year=2018, num_test_years=3, start_year=2005):
    data, _ = tennis_model.get_all_data(attributes, test_season=test_year-num_test_years+1, start_year=start_year)
    test_data, _ = tennis_model.get_all_data(attributes, tournament=test_tournament, test_season=test_year+1, start_year=test_year+1-num_test_years)
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


def load_data(start_year, test_year, num_test_years, model=None, spread_model=None):
    attributes = list(tennis_model.all_attributes)
    data, test_data = load_outcome_predictions_and_actuals(attributes, test_year=test_year, num_test_years=num_test_years,
                                                           start_year=start_year, model=model, spread_model=spread_model)
    # merge betting data in memory
    betting_sites = ['Bovada',
                     '5Dimes',  # 5Dimes available for money line usually (but not for spreads)
                     'BetOnline']
    betting_data = load_betting_data(betting_sites, test_year=test_year)
    test_data = pd.DataFrame.merge(
        test_data,
        betting_data,
        'inner',
        left_on=['year', 'player_id', 'opponent_id', 'tournament'],
        right_on=['year', 'team1', 'team2', 'tournament'],
        validate='1:1'
    )
    data = pd.DataFrame.merge(
        data,
        betting_data,
        'inner',
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
    model_parameters['epsilon'] = 0.1 + float(np.random.rand(1))*0.8
    model_parameters['bayes_model_percent'] = float(np.random.rand(1))
    model_parameters['min_odds'] = 0.05 + float(np.random.rand(1))*0.4
    model_parameters['max_odds'] = float(np.random.rand(1))*0.55+0.4
    return model_parameters


class MoneyLineSolution(Solution):
    def __init__(self, parameters):
        self.parameters = parameters.copy()

    def score(self, data):
        # run simulation
        return test(data, self.parameters)

    def mutate(self):
        mutated_parameters = self.parameters.copy()
        second_copies = new_random_parameters()
        for k in second_copies:
            if np.random.rand(1) < 0.25:
                mutated_parameters[k] = (second_copies[k]+mutated_parameters[k])/2.
        return MoneyLineSolution(mutated_parameters)

    def cross_over(self, other):
        cross_parameters = self.parameters.copy()
        for k in cross_parameters:
            if np.random.rand(1) < 0.5:
                cross_parameters[k] = other.parameters[k]
        return MoneyLineSolution(cross_parameters)


def solution_generator():
    return MoneyLineSolution(new_random_parameters())


def sample2d(array, seed, max_samples):
    i = seed % max_samples
    if i == 0:
        np.random.seed(seed)
        np.random.shuffle(array)
        return array[0:int(array.shape[0]/max_samples)]
    else:
        interval = int(array.shape[0]/max_samples)
        start = interval * i
        end = interval * i + interval
        return array[start:end]


def test(all_predictions, model_parameters):
    bayes_model_percent = model_parameters['bayes_model_percent']
    logit_percent = 1.0 - bayes_model_percent
    price_str = 'max_price'
    total = logit_percent * len(all_predictions[0]) + bayes_model_percent * len(all_predictions[1])
    avg_predictions = np.vstack([np.vstack(all_predictions[0]) * logit_percent,
                                 np.vstack(all_predictions[1]) * bayes_model_percent]).sum(0) / total

    _, _, _, avg_error = tennis_model.score_predictions(avg_predictions, test_data[y_str])
    test_return, num_bets = simulate_money_line(lambda j: avg_predictions[j], lambda j: test_data[y_str].iloc[j],
                                                bet_func(model_parameters['epsilon'],model_parameters), test_data,
                                                price_str, 1, sampling=0, shuffle=True, verbose=False)
    print('Avg model: ')
    print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:', to_percentage(avg_error),
          )  # ' Test years:', num_test_years, ' Test year:', test_year)
    if test_return > 0:
        score = max(0, test_return * float(num_bets - 10))
    else:
        score = test_return - float(np.log(1+num_bets))
    print('Score: ', score)
    print(model_parameters)
    print('---------------------------------------------------------')
    return score


def predict(data, test_data):
    #for num_test_years in [1]:
    #    for test_year in [2016, 2017, 2018]:
    graph = False
    lr = lambda: LogisticRegression()
    rf = lambda: RandomForestClassifier(n_estimators=300, max_depth=10)
    nb = lambda: GaussianNB()
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    X_train = np.array(data[betting_input_attributes].iloc[:, :])
    y_train = np.array(data[y_str].iloc[:]).flatten()
    X_test = np.array(test_data[betting_input_attributes].iloc[:, :])
    y_test = np.array(test_data[y_str].iloc[:]).flatten()
    all_predictions = []
    for _model, name in [
                    (lr, 'Logit Regression'),
                    (nb, 'Naive Bayes'),
                    #(rf, 'Random Forest'),
                ]:
        print('with Betting Model: ', name)
        #print("Shapes: ", X_train.shape, X_test.shape)
        model_predictions = []
        all_predictions.append(model_predictions)
        for i in range(1):
            model = _model()
            X_train_sample = sample2d(X_train, i, 1)
            y_train_sample = sample2d(y_train, i, 1)
            model.fit(X_train_sample, y_train_sample)
            #binary_correct, n, binary_percent, avg_error = test_model(model, X_test, y_test)
            #print('Correctly predicted: ' + str(binary_correct) + ' out of ' + str(n) +
            #      ' (' + to_percentage(binary_percent) + ')')
            prob_pos = predict_proba(model, X_test)
            model_predictions.append(prob_pos)
            if graph:
                fraction_of_positives, mean_predicted_value = \
                    calibration_curve(y_test, prob_pos, n_bins=10)
                ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                         label="%s" % (name,))

                ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                         histtype="step", lw=2)

            # actual returns on test data
            #test_return, num_bets = simulate_money_line(lambda j: prob_pos[j], lambda j: test_data.iloc[j]['y'],
            #                                  bet_func(model_parameters['model_to_epsilon'][name], model_parameters), test_data,
            #                                  price_str, num_tests, sampling=0, shuffle=True)
            #print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:', to_percentage(avg_error), ' Test years:', num_test_years, ' Test year:', test_year)



    if graph:
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')
        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)
        plt.tight_layout()
        plt.show()

    return all_predictions


if __name__ == '__main__':
    start_year = 2010
    num_test_years = 1
    test_year = 2018
    num_epochs = 50
    historical_model = load_outcome_model('Logistic')
    historical_spread_model = load_spread_model('Linear')
    data, test_data = load_data(start_year=start_year, num_test_years=num_test_years, test_year=test_year, model=historical_model, spread_model=historical_spread_model)
    all_predictions = predict(data, test_data)
    genetic_algorithm = GeneticAlgorithm(solution_generator=solution_generator)
    genetic_algorithm.fit(all_predictions, num_solutions=30, num_epochs=num_epochs)

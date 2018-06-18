from models.atp_tennis.TennisMatchOutcomeLogit import test_model, to_percentage
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
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
from models.simulation.Simulate import simulate_spread


betting_input_attributes = [
    #'prev_h2h2_wins_player',
    #'prev_h2h2_wins_opponent',
    #'prev_duration',
    #'opp_prev_duration',
    #'h2h_prior_win_percent',
    #'prev_year_prior_encounters',
    #'opp_prev_year_prior_encounters',
    #'tourney_hist_prior_encounters',
    #'opp_tourney_hist_prior_encounters',
    #'tiebreak_win_percent',
    #'opp_tiebreak_win_percent',
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
    'elo_score',
    'opp_elo_score',
    'grand_slam',
    'clay',
    'grass',
]

betting_only_attrs = [
    'odds1',
    'odds2',
    'spread1',
    'spread2',
    'spread_predictions',
    #'spread2',
    #'predictions'
]

y_str = 'beat_spread'
for attr in betting_only_attrs:
    if attr not in betting_input_attributes:
        betting_input_attributes.append(attr)

all_attributes = list(betting_input_attributes)
all_attributes.append(y_str)
meta_attributes = ['player_id', 'opponent_id', 'tournament', 'year']
for meta in meta_attributes:
    all_attributes.append(meta)


def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        prob_pos = model.predict_proba(X)[:, 1]
    else:  # use decision function
        prob_pos = model.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    return prob_pos


def load_outcome_predictions_and_actuals(model, spread_model, attributes, test_year=2018, num_test_years=3, start_year=2005):
    data, _ = tennis_model.get_all_data(attributes,include_spread=True,test_season=test_year-num_test_years+1, start_year=start_year)
    test_data, _ = tennis_model.get_all_data(attributes,include_spread=True,test_season=test_year+1, start_year=test_year+1-num_test_years)
    labels = data[1]
    spreads = labels[1]
    labels = labels[0]
    data = data[0]
    test_labels = test_data[1]
    test_spreads = test_labels[1]
    test_labels = test_labels[0]
    test_data = test_data[0]
    X = np.array(data[outcome_input_attributes])
    X_test = np.array(test_data[outcome_input_attributes])
    X_spread = np.array(data[spread_input_attributes])
    X_test_spread = np.array(test_data[spread_input_attributes])
    data['predictions'] = predict_proba(model, X)
    test_data['predictions'] = predict_proba(model, X_test)
    data['spread_predictions'] = spread_model.predict(X_spread)
    test_data['spread_predictions'] = spread_model.predict(X_test_spread)
    data['actual'] = labels
    data['spread_actual'] = spreads
    test_data['actual'] = test_labels
    test_data['spread_actual'] = test_spreads
    return data, test_data


def load_betting_data(betting_sites, test_year=2018):
    conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
    betting_data = pd.read_sql('''
        select year,tournament,team1,team2,
        book_name,
        price1,
        price2,
        spread1,
        spread2,
        odds1,
        odds2,
        betting_date
        from atp_tennis_betting_link_spread 
        where year<={{YEAR}} and book_name in ({{BOOK_NAMES}})
    '''.replace('{{YEAR}}', str(test_year)).replace('{{BOOK_NAMES}}', '\''+'\',\''.join(betting_sites)+'\''), conn)
    return betting_data


def extract_beat_spread_binary(spreads, spread_actuals):
    res = []
    ties = 0
    for i in range(len(spreads)):
        spread = spreads[i]
        spread_actual = spread_actuals[i]
        if spread >= 0:
            if spread == -spread_actual:
                if np.random.rand(1) > 0.5:
                    r = 1.0
                else:
                    r = 0.0
                ties += 1
            elif spread > -spread_actual:
                r = 1.0
            else:
                r = 0.0
        else:
            if -spread == spread_actual:
                if np.random.rand(1) > 0.5:
                    r = 1.0
                else:
                    r = 0.0
                ties += 1
            elif -spread < spread_actual:
                r = 1.0
            else:
                r = 0.0
        res.append(r)
    print('Num ties: ', ties, 'out of', len(res))
    return res


def load_data(model, spread_model, start_year, test_year, num_test_years):
    attributes = list(tennis_model.all_attributes)
    if 'spread' not in attributes:
        attributes.append('spread')
    for attr in betting_input_attributes:
        if attr not in attributes and attr not in betting_only_attrs:
            attributes.append(attr)
    data, test_data = load_outcome_predictions_and_actuals(model, spread_model, attributes, test_year=test_year, num_test_years=num_test_years,
                                                               start_year=start_year)
    betting_sites = ['Bovada', 'BetOnline']
    betting_data = load_betting_data(betting_sites, test_year=test_year)
    test_data = pd.DataFrame.merge(
        test_data,
        betting_data,
        'inner',
        left_on=['year', 'player_id', 'opponent_id', 'tournament'],
        right_on=['year', 'team1', 'team2', 'tournament'])
    data = pd.DataFrame.merge(
        data,
        betting_data,
        'inner',
        left_on=['year', 'player_id', 'opponent_id', 'tournament'],
        right_on=['year', 'team1', 'team2', 'tournament'])

    data[y_str] = extract_beat_spread_binary(spreads=data['spread1'], spread_actuals=data['spread_actual'])
    test_data[y_str] = extract_beat_spread_binary(spreads=test_data['spread1'], spread_actuals=test_data['spread_actual'])
    data = data.sort_values(by=['betting_date'], inplace=False, ascending=True, kind='mergesort')
    test_data = test_data.sort_values(by=['betting_date'], inplace=False, ascending=True, kind='mergesort')
    data.reset_index(drop=True)
    test_data.reset_index(drop=True)
    return data, test_data


def bet_func(epsilon):
    def bet_func_helper(price, odds, spread, prediction):
        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)
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
        if expectation > epsilon:
            return 1. + expectation
        else:
            return 0
    return bet_func_helper


if __name__ == '__main__':
    model_to_epsilon = {
        'Logit Regression': 0.15,
        'Naive Bayes': 1.0,
        'Random Forest': 0.20,
        'Average': 0.05,
        #'Support Vector': 0.3
    }
    test_year = 2018
    start_year = 2011
    num_tests = 5
    num_test_years = 2
    graph = False
    all_predictions = []
    for outcome_model_name in ['Logistic', 'Naive Bayes']:
        outcome_model = load_outcome_model(outcome_model_name)
        spread_model = load_spread_model('Linear')
        data, test_data = load_data(outcome_model, spread_model, start_year=start_year, num_test_years=num_test_years, test_year=test_year)
        lr = LogisticRegression()
        svm = LinearSVC()
        rf = RandomForestClassifier(n_estimators=200)
        nb = GaussianNB()
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for model, name in [
                        (lr, 'Logit Regression'),
                        #(svm, 'Support Vector'),
                        #(nb, 'Naive Bayes'),
                        (rf, 'Random Forest'),
                    ]:
            print('Using outcome model:', outcome_model_name, 'with Betting Model: ', name)
            X_train = np.array(data[betting_input_attributes])
            y_train = np.array(data[y_str]).flatten()
            X_test = np.array(test_data[betting_input_attributes])
            y_test = np.array(test_data[y_str]).flatten()
            #print("Shapes: ", X_train.shape, X_test.shape)
            model.fit(X_train, y_train)
            binary_correct, n, binary_percent, avg_error = test_model(model, X_test, y_test)
            #print('Correctly predicted: ' + str(binary_correct) + ' out of ' + str(n) +
            #      ' (' + to_percentage(binary_percent) + ')')
            prob_pos = predict_proba(model, X_test)
            all_predictions.append(prob_pos)
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_test, prob_pos, n_bins=10)

            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                     label="%s" % (name,))

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                     histtype="step", lw=2)

            parameters = dict()
            parameters['max_loss_percent'] = 0.05
            test_return, num_bets = simulate_spread(lambda j: prob_pos[j], lambda j: test_data.iloc[j]['actual'], lambda j: test_data.iloc[j]['spread_actual'], lambda _: None,
                                              bet_func(model_to_epsilon[name]), test_data, parameters,
                                              'price', num_tests, sampling=0, shuffle=False)
            print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:', to_percentage(avg_error), ' Test years:', num_test_years)
            print('---------------------------------------------------------')

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

    avg_predictions = np.vstack(all_predictions).mean(0)
    _, _, _, avg_error = tennis_model.score_predictions(avg_predictions, test_data['actual'])
    test_return, num_bets = simulate_spread(lambda j: avg_predictions[j], lambda j: test_data.iloc[j]['actual'],
                                            lambda j: test_data.iloc[j]['spread_actual'], lambda _: None,
                                            bet_func(model_to_epsilon['Average']), test_data, parameters,
                                            'price', num_tests, sampling=0, shuffle=False, verbose=False)
    print('Avg model')
    print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:', to_percentage(avg_error),
          ' Test years:', num_test_years)
    print('---------------------------------------------------------')


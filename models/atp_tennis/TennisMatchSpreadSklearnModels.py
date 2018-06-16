from models.atp_tennis.TennisMatchOutcomeLogit import test_model, to_percentage
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVR
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
from models.atp_tennis.TennisMatchOutcomeSklearnModels import load_model, save_model
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from models.atp_tennis.TennisMatchOutcomeLogit import input_attributes as outcome_input_attributes
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from models.simulation.Simulate import simulate_spread


betting_input_attributes = [
    #'prev_h2h2_wins_player',
    #'prev_h2h2_wins_opponent',
    'h2h_prior_win_percent',
    'prev_year_prior_encounters',
    'opp_prev_year_prior_encounters',
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
    'predictions'
]

y_str = 'returns'
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


def load_outcome_predictions_and_actuals(model, attributes, test_year=2018, num_test_years=3, start_year=2005):
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
    data['predictions'] = predict_proba(model, X)
    test_data['predictions'] = predict_proba(model, X_test)
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
        odds2
        from atp_tennis_betting_link_spread 
        where year<={{YEAR}} and book_name in ({{BOOK_NAMES}})
    '''.replace('{{YEAR}}', str(test_year)).replace('{{BOOK_NAMES}}', '\''+'\',\''.join(betting_sites)+'\''), conn)
    return betting_data


def load_data(model, start_year, test_year, num_test_years):
    attributes = list(tennis_model.all_attributes)
    if 'spread' not in attributes:
        attributes.append('spread')
    for attr in betting_input_attributes:
        if attr not in attributes and attr not in betting_only_attrs:
            attributes.append(attr)
    data, test_data = load_outcome_predictions_and_actuals(model, attributes, test_year=test_year, num_test_years=num_test_years,
                                                           start_year=start_year)
    betting_sites = ['Bovada', '5Dimes', 'BetOnline']
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
    data[y_str] = data['actual']
    test_data[y_str] = test_data['actual']
    return data, test_data


if __name__ == '__main__':
    test_year = 2018
    start_year = 2006
    num_test_years = 2
    for outcome_model_name in ['Logistic', 'Naive Bayes']:
        outcome_model = load_model(outcome_model_name)
        data, test_data = load_data(outcome_model, start_year=start_year, num_test_years=num_test_years, test_year=test_year)
        lr = LogisticRegression()
        svm = LinearSVC()
        rf = RandomForestClassifier(n_estimators=500)
        nb = GaussianNB()
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for model, name in [
                        (lr, 'Logit Regression'),
                        (svm, 'Support Vector'),
                        (nb, 'Naive Bayes'),
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
            print('Average error: ', to_percentage(avg_error))
            prob_pos = predict_proba(model, X_test)
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_test, prob_pos, n_bins=10)

            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                     label="%s" % (name,))

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                     histtype="step", lw=2)

            parameters = {}
            parameters['max_loss_percent'] = 0.05


            def bet_func(price, odds, prediction):
                if 0 > prediction or prediction > 1:
                    print('Invalid prediction: ', prediction)
                    exit(1)
                if price > 0:
                    expectation_implied = odds * price + (1.-odds) * -100.
                    expectation = prediction * price + (1.-prediction) * -100.
                else:
                    expectation_implied = odds * 100. + (1.-odds) * price
                    expectation = prediction * 100. + (1.-prediction) * price
                expectation /= 100.
                expectation_implied /= 100.
                # print('Expectation:', expectation, ' Implied: ', expectation_implied)
                return expectation > 1.
            print('Test prediction shape: ', prob_pos.shape)
            test_return, num_bets = simulate_spread(lambda j: prob_pos[j], lambda j: test_data['actual'][j], lambda j: test_data['spread_actual'][j], lambda _: None,
                                              bet_func, test_data, parameters,
                                              'price', 1)
            print('Final test return:', test_return, ' Num bets:', num_bets)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        plt.show()


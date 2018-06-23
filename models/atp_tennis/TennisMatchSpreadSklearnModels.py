from models.atp_tennis.TennisMatchOutcomeLogit import test_model, to_percentage
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
from models.atp_tennis.TennisMatchOutcomeSklearnModels import load_outcome_model, load_spread_model
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from models.simulation.Simulate import simulate_spread
from models.atp_tennis.TennisMatchMoneyLineSklearnModels import sample2d, load_outcome_predictions_and_actuals, spread_input_attributes


betting_input_attributes = list(spread_input_attributes)

betting_only_attributes = [
    'spread1',
    'spread2',
    'odds1',
    'odds2'
]
for attr in betting_only_attributes:
    betting_input_attributes.append(attr)

all_attributes = list(betting_input_attributes)
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
    #print('Num ties: ', ties, 'out of', len(res))
    return res


def load_data(start_year, test_year, num_test_years):
    attributes = list(tennis_model.all_attributes)
    if 'spread' not in attributes:
        attributes.append('spread')
    data, test_data = load_outcome_predictions_and_actuals(attributes, test_year=test_year, num_test_years=num_test_years,
                                                               start_year=start_year)
    betting_sites = ['Bovada', 'BetOnline']
    betting_data = load_betting_data(betting_sites, test_year=test_year)
    #print('pre headers: ', test_data.columns)
    test_data = pd.DataFrame.merge(
        test_data,
        betting_data,
        'inner',
        left_on=['year', 'player_id', 'opponent_id', 'tournament'],
        right_on=['year', 'team1', 'team2', 'tournament'],
        validate='1:m'
    )
    #print('post headers: ', test_data.columns)
    data = pd.DataFrame.merge(
        data,
        betting_data,
        'inner',
        left_on=['year', 'player_id', 'opponent_id', 'tournament'],
        right_on=['year', 'team1', 'team2', 'tournament'],
        validate='1:m'
    )
    data = data.assign(beat_spread=pd.Series(extract_beat_spread_binary(spreads=data['spread1'].iloc[:], spread_actuals=data['spread'].iloc[:])).values)
    test_data = test_data.assign(beat_spread=pd.Series(extract_beat_spread_binary(spreads=test_data['spread1'].iloc[:], spread_actuals=test_data['spread'].iloc[:])).values)
    data.sort_values(by=['betting_date'], inplace=True, ascending=True, kind='mergesort')
    test_data.sort_values(by=['betting_date'], inplace=True, ascending=True, kind='mergesort')
    #data.reset_index(drop=True, inplace=True)
    #test_data.reset_index(drop=True, inplace=True)
    return data, test_data


def bet_func(epsilon):
    def bet_func_helper(price, odds, spread, prediction):
        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)
        if odds < 0.25 or odds > 0.65:
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
        if expectation > epsilon:
            return 1. + expectation
        else:
            return 0
    return bet_func_helper


if __name__ == '__main__':
    model_to_epsilon = {
        'Logit Regression': 0.20,
        'Naive Bayes': 0.5,
        #'Random Forest': 0.4,
        'Average': 0.4,
        #'Support Vector': 0.3
    }
    for num_test_years in [1, 2]:
        for test_year in [2016, 2017, 2018]:
            start_year = 2011
            graph = False
            all_predictions = []
            data, test_data = load_data(start_year=start_year, num_test_years=num_test_years, test_year=test_year)
            lr = lambda: LogisticRegression()
            svm = lambda: LinearSVC()
            rf = lambda: RandomForestClassifier(n_estimators=200)
            nb = lambda: GaussianNB()
            plt.figure(figsize=(10, 10))
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            X_train = np.array(data[betting_input_attributes].iloc[:, :])
            y_train = np.array(data['beat_spread'].iloc[:]).flatten()
            X_test = np.array(test_data[betting_input_attributes].iloc[:, :])
            y_test = np.array(test_data['beat_spread'].iloc[:]).flatten()
            for _model, name in [
                            (lr, 'Logit Regression'),
                            #(svm, 'Support Vector'),
                            (nb, 'Naive Bayes'),
                            #(rf, 'Random Forest'),
                        ]:
                print('With betting model: ', name)
                model_predictions = []
                all_predictions.append(model_predictions)
                for i in range(50):
                    model = _model()
                    X_train_sample = sample2d(X_train, i, 2)
                    y_train_sample = sample2d(y_train, i, 2)
                    #print("Shapes: ", X_train.shape, X_test.shape)
                    model.fit(X_train_sample, y_train_sample)
                    binary_correct, n, binary_percent, avg_error = test_model(model, X_test, y_test)
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

                    #test_return, num_bets = simulate_spread(lambda j: prob_pos[j], lambda j: test_data['spread'].iloc[j],
                    #                                  bet_func(model_to_epsilon[name]), test_data,
                    #                                  'price', 2, sampling=0, shuffle=True)
                    #print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:', to_percentage(avg_error), ' Test years:', num_test_years, ' Year:', test_year)
                    #print('---------------------------------------------------------')

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

            for bayes_model_percent in [0.6, 0.7, 0.8, 1.]:
                for epsilon in [0.2, 0.3, 0.4, 0.5, 0.6]:
                    print('Avg Model ->  Bayes Percentage:', bayes_model_percent, ' Epsilon:', epsilon)
                    logit_percent = 1.0 - bayes_model_percent
                    avg_predictions = np.vstack([np.vstack(all_predictions[0]) * logit_percent,
                                                 np.vstack(all_predictions[1]) * bayes_model_percent]).mean(0)
                    _, _, _, avg_error = tennis_model.score_predictions(avg_predictions, test_data['spread'].iloc[:])
                    test_return, num_bets = simulate_spread(lambda j: avg_predictions[j], lambda j: test_data['spread'].iloc[j],
                                                            bet_func(epsilon), test_data,
                                                            'price', 2, sampling=0, shuffle=True, verbose=False)

                    print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:', to_percentage(avg_error),
                          ' Test years:', num_test_years, ' Year:', test_year)
                    print('---------------------------------------------------------')


from models.atp_tennis.TennisMatchOutcomeLogit import test_model, to_percentage
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
from models.atp_tennis.TennisMatchOutcomeSklearnModels import load_model, save_model
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from models.atp_tennis.TennisMatchOutcomeLogit import input_attributes as outcome_input_attributes
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from models.simulation.Simulate import simulate_money_line

betting_input_attributes = [
    'prev_h2h2_wins_player',
    'prev_h2h2_wins_opponent',
    'h2h_prior_win_percent',
    'prev_year_prior_encounters',
    'opp_prev_year_prior_encounters',
    'tourney_hist_prior_encounters',
    'opp_tourney_hist_prior_encounters',
    'tiebreak_win_percent',
    'opp_tiebreak_win_percent',
    'surface_experience',
    'opp_surface_experience',
    'experience',
    'opp_experience',
    'age',
    'opp_age',
    'lefty',
    'opp_lefty',
    'weight',
    'opp_weight',
    'height',
    'opp_height',
    'elo_score',
    'opp_elo_score'
]

betting_only_attrs = [
    #'max_price1',
    #'max_price2',
    #'predictions'
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


def extract_returns(data,labels):
    returns = []
    parameters = {}
    parameters['max_loss_percent'] = 0.1
    parameters['betting_epsilon1'] = 0.0
    parameters['betting_epsilon2'] = 0.0
    parameters['max_price_plus'] = 800
    parameters['max_price_minus'] = -800

    def bet_func(price, odds, prediction):
        return prediction > odds

    for i in range(len(labels)):
        return1 = simulate_money_line(lambda i: data['predictions'][i], lambda i: labels[i], lambda _: None,
                                      bet_func, data[i:i+1], parameters,
                                      'max_price', 1, sampling=0)
        returns.append(return1)
    return returns


def load_outcome_predictions_and_actuals(model, attributes, test_year=2018, num_test_years=1, start_year=2005):
    data, _ = tennis_model.get_all_data(attributes,test_season=test_year-num_test_years+1, start_year=start_year)
    test_data, _ = tennis_model.get_all_data(attributes,test_season=test_year+1, start_year=test_year+1-num_test_years)
    labels = data[1]
    data = data[0]
    test_labels = test_data[1]
    test_data = test_data[0]
    X = np.array(data[outcome_input_attributes])
    X_test = np.array(test_data[outcome_input_attributes])
    y = np.array(labels).flatten()
    model.fit(X, y)
    avg_error = test_model(model, X_test, test_labels)
    print('Average error: ', avg_error)
    data['predictions'] = model.predict(X)
    test_data['predictions'] = model.predict(X_test)
    data['actual'] = labels
    test_data['actual'] = test_labels
    return data, test_data


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
         where year<={{YEAR}} and book_name in ({{BOOK_NAMES}})
         group by year,tournament,team1,team2
     '''.replace('{{YEAR}}', str(test_year)).replace('{{BOOK_NAMES}}', '\'' + '\',\''.join(betting_sites) + '\''), conn)
    return betting_data


def load_data(model, start_year, test_year):
    attributes = tennis_model.all_attributes
    for attr in betting_input_attributes:
        if attr not in attributes and attr not in betting_only_attrs:
            attributes.append(attr)
    data, test_data = load_outcome_predictions_and_actuals(model, attributes, test_year=test_year, num_test_years=1, start_year=start_year)
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
    data[y_str] = extract_returns(data, data['actual'])
    test_data[y_str] = extract_returns(test_data, test_data['actual'])
    return data, test_data


if __name__ == '__main__':
    test_year = 2018
    start_year = 2006
    for outcome_model_name in ['Logistic', 'Naive Bayes']:
        outcome_model = load_model(outcome_model_name)
        data, test_data = load_data(outcome_model, start_year=start_year, test_year=test_year)
        print('Using outcome model: ', outcome_model_name)
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=500)
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        for model, name in [
                        (lr, 'Linear Regression'),
                        (rf, 'Random Forest Regressor'),
                    ]:
            X_train = np.array(data[betting_input_attributes])
            y_train = np.array(data[y_str]).flatten()
            X_test = np.array(test_data[betting_input_attributes])
            y_test = np.array(test_data[y_str]).flatten()

            print("Shapes: ", X_train.shape, X_test.shape)

            model.fit(X_train, y_train)
            n, avg_error = test_model(model, X_test, y_test, include_binary=False)

            print('Average error: ', avg_error)

            prediction = model.predict(X_test)
            errors = prediction - y_test
            ax1.plot(list(range(len(errors))), errors, "s-",
                     label="%s" % (name,))

        ax1.set_ylabel("Error")
        ax1.set_ylim([min(errors)-0.5, max(errors)+0.5])
        ax1.legend(loc="lower right")
        ax1.set_title('Errors')

        plt.tight_layout()
        plt.show()


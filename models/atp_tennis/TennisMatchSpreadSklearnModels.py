from models.atp_tennis.TennisMatchOutcomeLogit import test_model, to_percentage
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
from models.atp_tennis.SpreadMonteCarlo import probability_beat_given_win, probability_beat_given_loss
from models.atp_tennis.TennisMatchOutcomeSklearnModels import load_outcome_model, load_spread_model
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from models.simulation.Simulate import simulate_money_line
from models.atp_tennis.TennisMatchMoneyLineSklearnModels import sample2d, load_outcome_predictions_and_actuals, spread_input_attributes


betting_input_attributes = [
    # 'mean_duration',
    # 'mean_opp_duration',
    # 'mean_return_points_made',
    # 'mean_opp_return_points_made',
    # 'mean_second_serve_points_made',
    # 'mean_opp_second_serve_points_made',
    'h2h_prior_win_percent',
    # 'prev_year_prior_encounters',
    # 'opp_prev_year_prior_encounters',
    # 'prev_year_avg_round',
    # 'opp_prev_year_avg_round',
    # 'opp_tourney_hist_avg_round',
    # 'tourney_hist_avg_round',
    # 'tourney_hist_prior_encounters',
    # 'opp_tourney_hist_prior_encounters',
    # 'mean_break_points_made',
    # 'mean_opp_break_points_made',
    # 'previous_tournament_round',
    # 'opp_previous_tournament_round',
    # 'tiebreak_win_percent',
    # 'opp_tiebreak_win_percent',
    # 'surface_experience',
    # 'opp_surface_experience',
    # 'experience',
    # 'opp_experience',
    # 'age',
    # 'opp_age',
    # 'height',
    # 'opp_height',
    # 'duration_prev_match',
    # 'opp_duration_prev_match',
    'elo_score',
    'opp_elo_score',
    'avg_games_per_set',
    'opp_avg_games_per_set',
    #'best_year',
    #'opp_best_year',
    'historical_avg_odds',
    'prev_odds',
    'opp_prev_odds',
    'underdog_wins',
    'opp_underdog_wins',
    'fave_wins',
    'opp_fave_wins'
]

betting_only_attributes = [
    #'probability_beat',
    'ml_odds_avg',
    'predictions',
    'spread_predictions'
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
        select s.year,s.tournament,s.team1,s.team2,
        s.book_name,
        s.price1,
        s.price2,
        s.spread1,
        s.spread2,
        s.odds1,
        s.odds2,
        s.betting_date,
        m.odds1 as ml_odds1,
        m.odds2 as ml_odds2,   
        (m.odds1+(1.0-m.odds2))/2.0 as ml_odds_avg,
        m.price1 as max_price1,
        m.price2 as max_price2   
        from atp_tennis_betting_link_spread  as s join
        atp_tennis_betting_link as m on 
            ((m.team1,m.team2,m.tournament,m.book_name,m.year)=(s.team1,s.team2,s.tournament,s.book_name,s.year))
        where s.year<={{YEAR}} and s.book_name in ({{BOOK_NAMES}})
        and s.spread1 = - s.spread2
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


def load_data(start_year, test_year, num_test_years, test_tournament=None, model=None, spread_model=None):
    attributes = list(tennis_model.all_attributes)
    if 'spread' not in attributes:
        attributes.append('spread')
    for attr in betting_input_attributes:
        if attr not in betting_only_attributes and attr not in attributes:
            attributes.append(attr)
    data, test_data = load_outcome_predictions_and_actuals(attributes, test_tournament=test_tournament, model=model, spread_model=spread_model, test_year=test_year, num_test_years=num_test_years,
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
    #data = data.assign(beat_spread=pd.Series(extract_beat_spread_binary(spreads=data['spread1'].iloc[:], spread_actuals=data['spread'].iloc[:])).values)
    #test_data = test_data.assign(beat_spread=pd.Series(extract_beat_spread_binary(spreads=test_data['spread1'].iloc[:], spread_actuals=test_data['spread'].iloc[:])).values)
    data = data.assign(probability_beat=pd.Series([probability_beat_given_win(x) for x in data['spread1'].iloc[:]]).values)
    test_data = test_data.assign(probability_beat=pd.Series([probability_beat_given_win(x) for x in test_data['spread1'].iloc[:]]).values)
    #data.sort_values(by=['betting_date'], inplace=True, ascending=True, kind='mergesort')
    #test_data.sort_values(by=['betting_date'], inplace=True, ascending=True, kind='mergesort')
    #data.reset_index(drop=True, inplace=True)
    #test_data.reset_index(drop=True, inplace=True)
    return data, test_data


alpha = 1.0
def bet_func(epsilon):
    def bet_func_helper(price, odds, prediction, row):
        prediction = prediction * alpha + (1.0 - alpha) * odds
        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)
        if odds < 0.35 or odds > 0.50:
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
        if expectation > epsilon * 2:
            return 1. + expectation
        else:
            return 0
    return bet_func_helper


def spread_bet_func(epsilon):
    def bet_func_helper(price, odds, spread, prediction, row, ml_bet_player, ml_bet_opp, ml_opp_odds):
        spread_prob = probability_beat_given_win(spread, row['grand_slam'] > 0.5)
        spread_prob_loss = probability_beat_given_loss(spread, row['grand_slam'] > 0.5)
        prediction = prediction * alpha + (1.0 - alpha) * odds
        #prediction = alpha * prediction + (1.0 - alpha) * spread_prob
        prediction = spread_prob * prediction + spread_prob_loss * (1.0-prediction)
        #odds = alpha * odds + (1.0 - alpha) * spread_prob
        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)

        if ml_bet_player > 0:
            return 0

        if odds < 0.475 or odds > 0.525:
            return 0

        if ml_bet_opp > 0.0 and ml_opp_odds < 0.42 and spread_prob > ml_opp_odds:
            return 1.1

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


def predict(data, test_data, graph=False, train=True, prediction_function=None):
    graph = False
    all_predictions = []
    lr = lambda: LogisticRegression()
    svm = lambda: LinearSVC()
    rf = lambda: RandomForestClassifier(n_estimators=50)
    nb = lambda: GaussianNB()
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    y_str = 'y'  # 'beat_spread'
    X_train = np.array(data[betting_input_attributes].iloc[:, :])
    y_train = np.array(data[y_str].iloc[:]).flatten()
    X_test = np.array(test_data[betting_input_attributes].iloc[:, :])
    y_test = np.array(test_data[y_str].iloc[:]).flatten()
    for _model, name in [
        (lr, 'Logit Regression'),
        # (svm, 'Support Vector'),
        (nb, 'Naive Bayes'),
        (rf, 'Random Forest'),
    ]:
        print('With betting model: ', name)
        model_predictions = []
        all_predictions.append(model_predictions)
        seed = int(np.random.randint(0, high=1000000, size=1)) * 2
        for i in range(50):
            model = _model()
            X_train_sample = sample2d(X_train, seed + i, 2)
            y_train_sample = sample2d(y_train, seed + i, 2)
            # print("Shapes: ", X_train.shape, X_test.shape)
            model.fit(X_train_sample, y_train_sample)
            binary_correct, n, binary_percent, avg_error = test_model(model, X_test, y_test)
            # print('Correctly predicted: ' + str(binary_correct) + ' out of ' + str(n) +
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

            # test_return, num_bets = simulate_spread(lambda j: prob_pos[j], lambda j: test_data['spread'].iloc[j],
            #                                  bet_func(model_to_epsilon[name]), test_data,
            #                                  'price', 2, sampling=0, shuffle=True)
            # print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:', to_percentage(avg_error), ' Test years:', num_test_years, ' Year:', test_year)
            # print('---------------------------------------------------------')

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

    predictions = []

    #train_params = [
    #    [0.75, [0.5,0.525,0.55]],
    #    [0.8, [0.55,0.575,0.6]],
    #    [0.85, [0.55,0.575,0.6]]
    #]

    # production params DO NOT CHANGE!
    #train_params = [
    #    [0.9, [0.6, 0.625, 0.65]],
    #    [0.925, [0.625, 0.65, 0.675]],
    #    [0.95, [0.625, 0.65, 0.675]]
    #]

    # dev parameters
    train_params = [
        #[0.1, [0.08, 0.1, 0.15]],
        #[0.3, [0.05, 0.10, 0.15]],
        #[0.3, [0.0, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08]],
        [0.8, 0.1, [0.025, 0.05, 0.1, 0.15, 0.25]],
        [0.85, 0.075, [0.025, 0.05, 0.1, 0.15, 0.25]],
        [0.9, 0.05, [0.025, 0.05, 0.1, 0.15, 0.25]],
        #[0.7, [0.1, 0.15, 0.20]],
        #[0.9, [0.25, 0.3, 0.35]],
    ]

    test_idx = 1

    if train:
        params = train_params
    else:
        test_params = [
            [train_params[test_idx][0], train_params[test_idx][1], [train_params[test_idx][2][0]]]
        ]
        params = test_params
        print("Test params: ", params)

    for bayes_model_percent, logit_percent, epsilons in params:
        for epsilon in epsilons:
            if train:
                variance = 0.0001
                bayes_model_percent = bayes_model_percent + float(np.random.randn(1) * variance)
                epsilon = epsilon + float(np.random.randn(1) * variance)
            print('Avg Model ->  Bayes Percentage:', bayes_model_percent, ' Epsilon:', epsilon, ' Alpha:', alpha)
            rf_model_percent = 1.0 - logit_percent - bayes_model_percent
            total = logit_percent * len(all_predictions[0]) + bayes_model_percent * len(all_predictions[1]) + rf_model_percent * len(all_predictions[2])
            avg_predictions = np.vstack([np.vstack(all_predictions[0]) * logit_percent,
                                         np.vstack(all_predictions[1]) * bayes_model_percent,
                                         np.vstack(all_predictions[2]) * rf_model_percent]).sum(0) / total
            predictions.append(avg_predictions)
            if prediction_function is not None:
                prediction_function(avg_predictions, epsilon)
    return predictions


def prediction_func(avg_predictions, epsilon):
    _, _, _, avg_error = tennis_model.score_predictions(avg_predictions, test_data['spread'].iloc[:])
    test_return, num_bets = simulate_money_line(lambda j: avg_predictions[j],
                                                lambda j: test_data['y'].iloc[j],
                                                lambda j: test_data['spread'].iloc[j],
                                                bet_func(epsilon),
                                                spread_bet_func(epsilon),
                                                test_data,
                                                'max_price', 'price', 1, sampling=0,
                                                shuffle=True, verbose=False)

    print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:',
          to_percentage(avg_error),
          ' Test years:', num_test_years, ' Year:', test_year)
    print('---------------------------------------------------------')


start_year = 2011
if __name__ == '__main__':
    historical_model = load_outcome_model('Logistic')
    historical_spread_model = load_spread_model('Linear')
    num_tests = 1
    for i in range(num_tests):
        print("TEST: ", i)
        for num_test_years in [1, ]:
            for test_year in [2017, 2018]:
                graph = False
                all_predictions = []
                data, test_data = load_data(start_year=start_year, num_test_years=num_test_years,
                                            test_year=test_year, model=historical_model, spread_model=historical_spread_model)
                avg_predictions = predict(data, test_data, prediction_function=prediction_func, graph=False, train=True)

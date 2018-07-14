from models.atp_tennis.TennisMatchOutcomeLogit import test_model, to_percentage
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
from models.atp_tennis.SpreadProbabilitiesByPlayer import spread_prob, total_sets_prob, total_games_prob, abs_probabilities_per_surface, abs_game_total_probabilities_per_surface, abs_set_total_probabilities_per_surface
from models.atp_tennis.TennisMatchOutcomeSklearnModels import load_outcome_model, load_spread_model
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from models.simulation.Simulate import simulate_money_line


totals_type_by_betting_site = {  # describes the totals type for each betting site
    'Bovada': 'Set',
    'BetOnline': 'Game',
    '5Dimes': 'Game',
}

betting_sites = list(totals_type_by_betting_site.keys())

betting_input_attributes = [
    #'h2h_prior_win_percent',
    #'historical_avg_odds',
    'prev_odds',
    'opp_prev_odds',
    'underdog_wins',
    'opp_underdog_wins',
    #'fave_wins',
    #'elo_score',
    #'opp_elo_score',
    #'opp_fave_wins',
]

betting_only_attributes = [
    'ml_odds_avg',
    'predictions',
    #'spread_predictions'
]

for attr in betting_only_attributes:
    betting_input_attributes.append(attr)


all_attributes = list(betting_input_attributes)
meta_attributes = ['player_id', 'opponent_id', 'tournament', 'year', 'ml_odds_avg']
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


def sample2d(array, seed, sample_percent):
    np.random.seed(seed)
    np.random.shuffle(array)
    return array[0:int(array.shape[0]*sample_percent)]


def load_betting_data(betting_sites, test_year=2018):
    conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
    betting_data = pd.read_sql('''
        select m.year,m.tournament,m.team1,m.team2,
        m.book_name,
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
        m.price2 as max_price2,
        t.price1 as totals_price1,
        t.price2 as totals_price2,
        t.over,
        t.under  
        from atp_tennis_betting_link as m 
        left outer join atp_tennis_betting_link_spread  as s
        on ((m.team1,m.team2,m.tournament,m.book_name,m.year)=(s.team1,s.team2,s.tournament,s.book_name,s.year)
            and s.spread1=-s.spread2)
        left outer join atp_tennis_betting_link_totals as t
        on ((m.team1,m.team2,m.tournament,m.book_name,m.year)=(t.team1,t.team2,t.tournament,t.book_name,t.year)
            and t.over=t.under)
        where m.year<={{YEAR}} and m.book_name in ({{BOOK_NAMES}})

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


def load_outcome_predictions_and_actuals(attributes, test_tournament=None, model=None, spread_model=None, slam_model=None, slam_spread_model=None, test_year=2018, num_test_years=3, start_year=2005):
    data, _ = tennis_model.get_all_data(attributes, test_season=test_year-num_test_years+1, start_year=start_year)
    test_data, _ = tennis_model.get_all_data(attributes, tournament=test_tournament, test_season=test_year+1, start_year=test_year+1-num_test_years)
    if model is not None and slam_model is not None:
        attrs = tennis_model.input_attributes0
        X = np.array(data[attrs].iloc[:, :])
        X_test = np.array(test_data[attrs].iloc[:, :])
        y_hat = predict_proba(model, X)
        y_hat_test = predict_proba(model, X_test)
        y_hat_slam = predict_proba(slam_model, X)
        y_hat_slam_test = predict_proba(slam_model, X_test)

        lam_rat = 0.75
        def lam(y, y_slam, slam):
            if slam > 0.5:
                return y_slam * lam_rat + y * (1.0 - lam_rat)
            else:
                return y * lam_rat + y_slam * (1.0 - lam_rat)

        data = data.assign(predictions=pd.Series([lam(y_hat[i],y_hat_slam[i],data['grand_slam'].iloc[i]) for i in range(data.shape[0])]).values)
        test_data = test_data.assign(predictions=pd.Series([lam(y_hat_test[i],y_hat_slam_test[i],test_data['grand_slam'].iloc[i]) for i in range(test_data.shape[0])]).values)

    if spread_model is not None and slam_spread_model is not None:
        X_spread = np.array(data[tennis_model.input_attributes_spread].iloc[:, :])
        X_test_spread = np.array(test_data[tennis_model.input_attributes_spread].iloc[:, :])
        y_hat = spread_model.predict(X_spread)
        y_hat_test = spread_model.predict(X_test_spread)
        y_hat_slam = slam_spread_model.predict(X_spread)
        y_hat_slam_test = slam_spread_model.predict(X_test_spread)

        def lam(y, y_slam, slam):
            if slam > 0.5:
                return y_slam
            else:
                return y

        data = data.assign(spread_predictions=pd.Series(
            [lam(y_hat[i], y_hat_slam[i], data['grand_slam'].iloc[i]) for i in range(data.shape[0])]).values)
        test_data = test_data.assign(spread_predictions=pd.Series(
            [lam(y_hat_test[i], y_hat_slam_test[i], test_data['grand_slam'].iloc[i]) for i in
             range(test_data.shape[0])]).values)

    return data, test_data


def load_data(start_year, test_year, num_test_years, test_tournament=None, model=None, slam_model=None,
              spread_model=None, slam_spread_model=None):
    attributes = list(tennis_model.all_attributes)
    if 'spread' not in attributes:
        attributes.append('spread')
    if 'totals' not in attributes:
        attributes.append('totals')

    for attr in betting_input_attributes:
        if attr not in betting_only_attributes and attr not in attributes:
            attributes.append(attr)
    data, test_data = load_outcome_predictions_and_actuals(attributes, test_tournament=test_tournament, model=model, slam_model=slam_model, spread_model=spread_model, slam_spread_model=slam_spread_model, test_year=test_year, num_test_years=num_test_years,
                                                               start_year=start_year)

    betting_data = load_betting_data(betting_sites, test_year=test_year)

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
    #data.sort_values(by=['betting_date'], inplace=True, ascending=True, kind='mergesort')
    #test_data.sort_values(by=['betting_date'], inplace=True, ascending=True, kind='mergesort')
    #data.reset_index(drop=True, inplace=True)
    #test_data.reset_index(drop=True, inplace=True)
    return data, test_data


alpha = 1.0
spread_cushion = 0.0
dont_bet_against_spread = {

}


def bet_func(epsilon, bet_ml=True):
    def bet_func_helper(price, odds, prediction, bet_row):
        if not bet_ml:
            return 0
        if (bet_row['grand_slam'] > 0.5 and bet_row['round_num'] < 2) or \
                (bet_row['grand_slam'] < 0.5 and bet_row['round_num'] < 3):
            return 0

        prediction = prediction * alpha + (1.0 - alpha) * odds
        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)
        if odds < 0.20 or odds > 0.55:
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


def totals_bet_func(epsilon, bet_totals=True):
    def bet_func_helper(price, odds, spread_prob_win, spread_prob_loss, prediction, row, ml_bet_player,
                        ml_bet_opp, ml_opp_odds):
        if not bet_totals:
            return 0
        prediction = prediction * alpha + (1.0 - alpha) * odds
        prediction = spread_prob_win * prediction + spread_prob_loss * (1.0 - prediction)

        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)

        if odds < 0.46 or odds > 0.540:
            return 0

        double_down_below = 0  # 0.35
        hedge_below = 0  # 0.45

        if double_down_below > 0:
            if ml_bet_player > 0 and (1.0 - ml_opp_odds) < double_down_below:
                return 1.05

        if hedge_below > 0:
            if ml_bet_opp > 0.0 and ml_opp_odds < hedge_below:
                return 1.05

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


def spread_bet_func(epsilon, bet_spread=True):
    def bet_func_helper(price, odds, spread_prob_win, spread_prob_loss, prediction, bet_row, ml_bet_player, ml_bet_opp, ml_opp_odds):
        if not bet_spread:
            return 0
        #if bet_row['grand_slam'] < 0.5:
        #    return 0

        prediction = prediction * alpha + (1.0 - alpha) * odds
        prediction = spread_prob_win * prediction + spread_prob_loss * (1.0-prediction)

        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)

        if odds < 0.45 or odds > 0.525:
            return 0

        double_down_below = 0  # 0.35
        hedge_below = 0  # 0.45

        if double_down_below > 0:
            if ml_bet_player > 0 and (1.0 - ml_opp_odds) < double_down_below:
                return 1.05

        if hedge_below > 0:
            if ml_bet_opp > 0.0 and ml_opp_odds < hedge_below:
                return 1.05

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
    all_predictions = []
    lr = lambda: LogisticRegression()
    svm = lambda: LinearSVC()
   # rf = lambda: RandomForestClassifier(n_estimators=300)
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
       # (rf, 'Random Forest'),
    ]:
        print('With betting model: ', name)
        model_predictions = []
        all_predictions.append(model_predictions)
        seed = int(np.random.randint(0, high=1000000, size=1)) * 2
        if name == 'Random Forest':
            model = _model()
            model.fit(X_train, y_train)
            prob_pos = predict_proba(model, X_test)
            model_predictions.append(prob_pos)
        else:
            for i in range(50):
                model = _model()
                ratio = 0.75
                X_train_sample = sample2d(X_train, seed + i, ratio)
                y_train_sample = sample2d(y_train, seed + i, ratio)
                model.fit(X_train_sample, y_train_sample)
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
        #[0.1, 0.5, [0.08, 0.1, 0.15]],
        #[0.3, 0.5, [0.05, 0.10, 0.15]],
        #[0.3, [0.0, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08]],
        [0.1, [0., 0.01, 0.025, 0.05, 0.1, 0.20]],
        [0.5, [0., 0.01, 0.025, 0.05, 0.1, 0.20]],
        [0.9, [0., 0.01, 0.025, 0.05, 0.1, 0.20]],
        #[0.7, [0.1, 0.15, 0.20]],
        #[0.9, [0.25, 0.3, 0.35]],
    ]

    test_idx = 1

    if train:
        params = train_params
    else:
        test_params = [
            [train_params[test_idx][0], [train_params[test_idx][1][0]]]
        ]
        params = test_params
        print("Test params: ", params)

    for bayes_model_percent, epsilons in params:
        for epsilon in epsilons:
            if train:
                variance = 0.0001
                bayes_model_percent = bayes_model_percent + float(np.random.randn(1) * variance)
                epsilon = epsilon + float(np.random.randn(1) * variance)
            print('Avg Model ->  Bayes Percentage:', bayes_model_percent, ' Epsilon:', epsilon, ' Alpha:', alpha)
           # rf_model_percent = 1.0 - logit_percent - bayes_model_percent
            logit_percent = 1.0 - bayes_model_percent
            total = logit_percent * len(all_predictions[0]) + bayes_model_percent * len(all_predictions[1]) #+ rf_model_percent * len(all_predictions[2])
            avg_predictions = np.vstack([np.vstack(all_predictions[0]) * logit_percent,
                                         np.vstack(all_predictions[1]) * bayes_model_percent
                                      #   np.vstack(all_predictions[2]) * rf_model_percent
                                        ]).sum(0) / total
            predictions.append(avg_predictions)
            if prediction_function is not None:
                prediction_function(avg_predictions, epsilon)
    return predictions


def decision_func(epsilon, bet_ml=True, bet_spread=True, bet_totals=True):
    ml_func = bet_func(epsilon, bet_ml=bet_ml)
    spread_func = spread_bet_func(epsilon, bet_spread=bet_spread)
    totals_func = totals_bet_func(epsilon, bet_totals=bet_totals)
    priors_spread = abs_probabilities_per_surface
    priors_set_totals = abs_set_total_probabilities_per_surface
    priors_game_totals = abs_game_total_probabilities_per_surface

    def check_player_for_spread(bet, opponent):
        if opponent in dont_bet_against_spread:
            return 0.
        return bet

    def decision_func_helper(ml_bet_option, spread_bet_option, totals_bet_option, bet_row, prediction):
        if (bet_row['grand_slam'] > 0.5 and bet_row['round_num'] < 1) or \
                (bet_row['grand_slam'] < 0.5 and bet_row['round_num'] < 2):
            return {
                'ml_bet1': 0,
                'ml_bet2': 0,
                'spread_bet1': 0,
                'spread_bet2': 0,
                'over_bet': 0,
                'under_bet': 0
            }
        spread_prob_win1 = spread_prob(bet_row['player_id'], bet_row['opponent_id'], bet_row['tournament'], bet_row['year'],
                                       spread_bet_option.spread1 - spread_cushion, bet_row['grand_slam'] > 0.5, priors_spread,
                                       bet_row['court_surface'], win=True)
        spread_prob_win2 = spread_prob(bet_row['opponent_id'], bet_row['player_id'], bet_row['tournament'], bet_row['year'],
                                       spread_bet_option.spread2 - spread_cushion, bet_row['grand_slam'] > 0.5, priors_spread,
                                       bet_row['court_surface'], win=True)
        spread_prob_loss1 = spread_prob(bet_row['player_id'], bet_row['opponent_id'], bet_row['tournament'], bet_row['year'],
                                        spread_bet_option.spread1 - spread_cushion, bet_row['grand_slam'] > 0.5, priors_spread,
                                        bet_row['court_surface'], win=False)
        spread_prob_loss2 = spread_prob(bet_row['opponent_id'], bet_row['player_id'], bet_row['tournament'], bet_row['year'],
                                        spread_bet_option.spread2 - spread_cushion, bet_row['grand_slam'] > 0.5, priors_spread,
                                        bet_row['court_surface'], win=False)
        if totals_type_by_betting_site[bet_row['book_name']] == 'Game':
            totals_prob_under_win = total_games_prob(bet_row['player_id'], bet_row['tournament'], bet_row['year'],
                                                 totals_bet_option.under, bet_row['grand_slam'] > 0.5, priors_game_totals,
                                                 bet_row['court_surface'], under=True, win=True)
            totals_prob_over_win = total_games_prob(bet_row['opponent_id'], bet_row['tournament'], bet_row['year'],
                                                totals_bet_option.over, bet_row['grand_slam'] > 0.5, priors_game_totals,
                                                bet_row['court_surface'], under=False, win=True)
            totals_prob_under_loss = total_games_prob(bet_row['player_id'], bet_row['tournament'], bet_row['year'],
                                                 totals_bet_option.under, bet_row['grand_slam'] > 0.5, priors_game_totals,
                                                 bet_row['court_surface'], under=True, win=False)
            totals_prob_over_loss = total_games_prob(bet_row['opponent_id'], bet_row['tournament'], bet_row['year'],
                                                totals_bet_option.over, bet_row['grand_slam'] > 0.5, priors_game_totals,
                                                bet_row['court_surface'], under=False, win=False)
        else:
            totals_prob_under_win = total_sets_prob(bet_row['player_id'], bet_row['tournament'], bet_row['year'],
                                                totals_bet_option.under, bet_row['grand_slam'] > 0.5, priors_set_totals,
                                                bet_row['court_surface'], win=True, under=True)
            totals_prob_over_win = total_sets_prob(bet_row['opponent_id'], bet_row['tournament'], bet_row['year'],
                                               totals_bet_option.over, bet_row['grand_slam'] > 0.5, priors_set_totals,
                                               bet_row['court_surface'], win=True, under=False)
            totals_prob_under_loss = total_sets_prob(bet_row['player_id'], bet_row['tournament'], bet_row['year'],
                                                totals_bet_option.under, bet_row['grand_slam'] > 0.5, priors_set_totals,
                                                bet_row['court_surface'], win=False, under=True)
            totals_prob_over_loss = total_sets_prob(bet_row['opponent_id'], bet_row['tournament'], bet_row['year'],
                                               totals_bet_option.over, bet_row['grand_slam'] > 0.5, priors_set_totals,
                                               bet_row['court_surface'], win=False, under=False)

        ml_bet1 = ml_func(ml_bet_option.max_price1, ml_bet_option.best_odds1, prediction, bet_row)
        ml_bet2 = ml_func(ml_bet_option.max_price2, ml_bet_option.best_odds2, 1.0 - prediction, bet_row)
        spread_bet1 = spread_func(spread_bet_option.max_price1, spread_bet_option.best_odds1, spread_prob_win1, spread_prob_loss1,
                                  prediction, bet_row, ml_bet1, ml_bet2, ml_bet_option.best_odds2)
        spread_bet2 = spread_func(spread_bet_option.max_price2, spread_bet_option.best_odds2, spread_prob_win2, spread_prob_loss2,
                                  1.0 - prediction, bet_row, ml_bet2, ml_bet1, ml_bet_option.best_odds1)
        over_bet = totals_func(totals_bet_option.max_price1, totals_bet_option.best_odds1, totals_prob_over_win, totals_prob_over_loss,
                               prediction, bet_row, ml_bet1, ml_bet2, ml_bet_option.best_odds2)
        under_bet = totals_func(totals_bet_option.max_price2, totals_bet_option.best_odds2, totals_prob_under_win, totals_prob_under_loss,
                                1.0 - prediction, bet_row, ml_bet2, ml_bet1, ml_bet_option.best_odds1)

        # check whether it's an amazing player in a grand slam
        #   basically, we don't want to bet against them...
        if bet_row['grand_slam'] > 0.5:
            spread_bet1 = check_player_for_spread(spread_bet1, bet_row['opponent_id'])
            spread_bet2 = check_player_for_spread(spread_bet2, bet_row['player_id'])

        return {
            'ml_bet1': ml_bet1,
            'ml_bet2': ml_bet2,
            'spread_bet1': spread_bet1,
            'spread_bet2': spread_bet2,
            'over_bet': over_bet,
            'under_bet': under_bet
        }
    return decision_func_helper


def prediction_func(bet_ml=True, bet_spread=True, bet_totals=True):
    def prediction_func_helper(avg_predictions, epsilon):
        _, _, _, avg_error = tennis_model.score_predictions(avg_predictions, test_data['spread'].iloc[:])

        def game_or_set_totals_func(i):
            if totals_type_by_betting_site[test_data['book_name'].iloc[i]] == 'Game':
                return test_data['totals'].iloc[i]
            else:
                return test_data['num_sets'].iloc[i]

        test_return, num_bets = simulate_money_line(lambda j: avg_predictions[j],
                                                    lambda j: test_data['y'].iloc[j],
                                                    lambda j: test_data['spread'].iloc[j],
                                                    lambda j: game_or_set_totals_func(j),
                                                    decision_func(epsilon, bet_ml=bet_ml, bet_spread=bet_spread,
                                                                  bet_totals=bet_totals),
                                                    test_data,
                                                    'max_price', 'price', 'totals_price', 1, sampling=0,
                                                    shuffle=True, verbose=False)

        print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:',
              to_percentage(avg_error),
              ' Test years:', num_test_years, ' Year:', test_year)
        print('---------------------------------------------------------')
    return prediction_func_helper


start_year = 2011
if __name__ == '__main__':
    historical_model = load_outcome_model('Logistic0')
    historical_spread_model = load_spread_model('Linear0')
    historical_model_slam = load_outcome_model('Logistic1')
    historical_spread_model_slam = load_spread_model('Linear1')
    num_tests = 1
    bet_spread = True
    bet_ml = True
    bet_totals = False
    for i in range(num_tests):
        print("TEST: ", i)
        for num_test_years in [1, ]:
            for test_year in [2018, 2017]:
                graph = False
                all_predictions = []
                data, test_data = load_data(start_year=start_year, num_test_years=num_test_years,
                                            test_year=test_year, model=historical_model, slam_model=historical_model_slam,
                                            spread_model=historical_spread_model, slam_spread_model=historical_spread_model_slam)
                avg_predictions = predict(data, test_data, prediction_function=prediction_func(bet_ml=bet_ml, bet_spread=bet_spread, bet_totals=bet_totals), graph=False, train=True)

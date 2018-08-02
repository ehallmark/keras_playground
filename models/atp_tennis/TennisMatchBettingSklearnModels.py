from models.atp_tennis.TennisMatchOutcomeLogit import test_model, to_percentage
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
from models.atp_tennis.SpreadProbabilitiesByPlayer import spread_prob, total_sets_prob, total_games_prob, abs_game_total_probabilities_per_surface, abs_set_total_probabilities_per_surface
from models.atp_tennis.TennisMatchOutcomeSklearnModels import load_outcome_model, load_spread_model
from models.atp_tennis.SpreadMonteCarlo import abs_probabilities_per_surface
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from models.simulation.Simulate import simulate_money_line
import datetime
import keras as k
from keras.optimizers import Adam
import models.atp_tennis.TennisMatchOutcomeNN as nn

totals_type_by_betting_site = {  # describes the totals type for each betting site
    'Bovada': 'Set',
    'BetOnline': 'Game',
    '5Dimes': 'Game',
    'OddsPortal': 'Game'
}

betting_sites = list(totals_type_by_betting_site.keys())

betting_input_attributes = [
    #'h2h_prior_win_percent',
    #'historical_avg_odds',
    'prev_odds',
    'opp_prev_odds',
    'underdog_wins',
    'opp_underdog_wins',
    'fave_wins',
    'opp_fave_wins',
    #'elo_score',
    #'opp_elo_score',
]

betting_only_attributes = [
    #'overall_odds_avg',
    #'ml_odds_avg',
    'predictions', #'_nn',
    #'spread_predictions',
]

for attr in betting_only_attributes:
    betting_input_attributes.append(attr)

betting_only_attributes.append('ml_odds_avg')
betting_only_attributes.append('overall_odds_avg')
betting_only_attributes.append('spread_odds_avg')


all_attributes = list(betting_input_attributes)
meta_attributes = ['player_id', 'opponent_id', 'tournament', 'year', 'ml_odds_avg']
for meta in meta_attributes:
    if meta not in all_attributes:
        all_attributes.append(meta)

for attr in nn.input_attributes:
    if attr not in all_attributes:
        all_attributes.append(attr)

# model_nn = k.models.load_model('tennis_match_keras_nn_v5.h5')
# model_nn.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])


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


def load_betting_data(betting_sites, test_year=datetime.date.today()):
    conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
    betting_data = pd.read_sql('''
        select m.start_date,m.tournament,m.team1,m.team2,
        m.book_name,
        s.price1,
        s.price2,
        s.spread1,
        s.spread2,
        s.odds1,
        s.odds2,
        (s.odds1+(1.0-s.odds2))/2.0 as spread_odds_avg,        
        coalesce(coalesce(m.betting_date,s.betting_date),t.betting_date) as betting_date,
        m.odds1 as ml_odds1,
        m.odds2 as ml_odds2,   
        (m.odds1+(1.0-m.odds2))/2.0 as ml_odds_avg,
        m.price1 as max_price1,
        m.price2 as max_price2,
        t.price1 as totals_price1,
        t.price2 as totals_price2,
        t.over,
        t.under,
        ml_overall_avg.avg_odds as overall_odds_avg  
        from atp_tennis_betting_link as m 
        left outer join atp_tennis_betting_link_spread  as s
        on ((m.team1,m.team2,m.tournament,m.book_id,m.start_date)=(s.team1,s.team2,s.tournament,s.book_id,s.start_date)
            and s.spread1=-s.spread2)
        left outer join atp_tennis_betting_link_totals as t
        on ((m.team1,m.team2,m.tournament,m.book_id,m.start_date)=(t.team1,t.team2,t.tournament,t.book_id,t.start_date)
            and t.over=t.under)
        left outer join atp_matches_money_line_average as ml_overall_avg
            on ((m.team1,m.team2,m.start_date,m.tournament)=(ml_overall_avg.player_id,ml_overall_avg.opponent_id,ml_overall_avg.start_date,ml_overall_avg.tournament))        
        where m.betting_date<='{{YEAR}}'::date + interval '30 days' and m.book_name in ({{BOOK_NAMES}})

    '''.replace('{{YEAR}}', str(test_year.strftime('%Y-%m-%d'))).replace('{{BOOK_NAMES}}', '\''+'\',\''.join(betting_sites)+'\''), conn)
    return betting_data


def extract_beat_spread_binary(spreads, spread_actuals):
    res = []
    ties = 0
    for i in range(len(spreads)):
        spread = spreads[i]
        spread_actual = spread_actuals[i]
        if np.isnan(spread_actual) or np.isnan(spread):
            r = np.nan
        else:
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


def load_outcome_predictions_and_actuals(attributes, test_tournament=None, models=None, spread_models=None, test_year=datetime.date.today(), num_test_years = 1, start_year='2005-01-01', masters_min=101, num_test_months=0):
    data, test_data = tennis_model.get_all_data(attributes, tournament=test_tournament, test_season=test_year.strftime('%Y-%m-%d'), num_test_years=num_test_years, start_year=start_year, masters_min=masters_min, num_test_months=num_test_months)
    if models is not None:
        attrs = tennis_model.input_attributes0
        X = np.array(data[attrs].iloc[:, :])
        X_test = np.array(test_data[attrs].iloc[:, :])
        predictions = {}
        predictions_test = {}
        for key, model in models.items():
            if key == 'FirstRound':
                first_round_attrs = list(attrs)
                if 'previous_games_total2' in first_round_attrs:
                    first_round_attrs.remove('previous_games_total2')
                if 'opp_previous_games_total2' in first_round_attrs:
                    first_round_attrs.remove('opp_previous_games_total2')
                if X.shape[0] > 0:
                    X_ = np.array(data[first_round_attrs].iloc[:, :])
                    y_hat = predict_proba(model, X_)
                    predictions[key] = y_hat
                if X_test.shape[0] > 0:
                    X_test_ = np.array(test_data[first_round_attrs].iloc[:, :])
                    y_hat_test = predict_proba(model, X_test_)
                    predictions_test[key] = y_hat_test
            else:
                if X.shape[0]>0:
                    y_hat = predict_proba(model, X)
                    predictions[key] = y_hat
                if X_test.shape[0] > 0:
                    y_hat_test = predict_proba(model, X_test)
                    predictions_test[key] = y_hat_test

        c1 = 0.70
        c2 = 0.20
        c3 = 0.10
        def lam(i, row, predictions):
            rank = row['tournament_rank']
            if rank >= 2000:
                y = predictions['2000']
            elif rank == 1000:
                y = predictions['1000']
            elif rank == 100:
                y = predictions['100']
            elif rank == 25:
                y = predictions['25']
            else:
                y = predictions['500']

            if row['court_surface']=='Clay':
                y2 = predictions['Clay']
            elif row['court_surface']=='Grass':
                y2 = predictions['Grass']
            else:
                y2 = predictions['Hard']

            if row['first_round'] > 0.5:
                y3 = predictions['FirstRound']
            else:
                y3 = predictions['OtherRound']

            return float(c1 * y[i] + c2 * y2[i] + c3 * y3[i])

        # data = data.assign(predictions_nn=pd.Series([y_nn[i] for i in range(data.shape[0])]).values)
        # test_data = test_data.assign(predictions_nn=pd.Series([y_nn_test[i] for i in range(test_data.shape[0])]).values)
        data = data.assign(predictions=pd.Series([lam(i,data.iloc[i], predictions) for i in range(data.shape[0])]).values)
        test_data = test_data.assign(predictions=pd.Series([lam(i,test_data.iloc[i], predictions_test) for i in range(test_data.shape[0])]).values)

    return data, test_data


def load_data(start_year, test_year, num_test_years, test_tournament=None, models=None, spread_models=None, masters_min=101, num_test_months=0):
    attributes = list(tennis_model.all_attributes)
    if 'spread' not in attributes:
        attributes.append('spread')
    if 'totals' not in attributes:
        attributes.append('totals')

    for attr in betting_input_attributes:
        if attr not in betting_only_attributes and attr not in attributes:
            attributes.append(attr)
    for attr in all_attributes:
        if attr not in betting_only_attributes and attr not in attributes:
            attributes.append(attr)
    data, test_data = load_outcome_predictions_and_actuals(attributes, test_tournament=test_tournament, models=models, spread_models=spread_models, test_year=test_year, num_test_years=num_test_years,
                                                               start_year=start_year, masters_min=masters_min, num_test_months=num_test_months)

    betting_data = load_betting_data(betting_sites, test_year=test_year)

    test_data = pd.DataFrame.merge(
        test_data,
        betting_data,
        'inner',
        left_on=['start_date', 'player_id', 'opponent_id', 'tournament'],
        right_on=['start_date', 'team1', 'team2', 'tournament'],
        validate='1:m'
    )
    #print('post headers: ', test_data.columns)
    data = pd.DataFrame.merge(
        data,
        betting_data,
        'inner',
        left_on=['start_date', 'player_id', 'opponent_id', 'tournament'],
        right_on=['start_date', 'team1', 'team2', 'tournament'],
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
        if bet_row['first_round'] > 0.5 and bet_row['tournament_rank'] > 1000:
            return 0

        alpha_odds = bet_row['overall_odds_avg']
        prediction = prediction * alpha + (1.0 - alpha) * alpha_odds
        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)

        if bet_row['itf'] > 0.5:
            min_odds = 0.20
            max_odds = 0.525
            epsilon_real = epsilon
        elif bet_row['challenger'] > 0.5:
            min_odds = 0.20
            max_odds = 0.525
            epsilon_real = epsilon
        else:
            min_odds = 0.20
            max_odds = 0.525
            epsilon_real = epsilon

        if odds < min_odds or odds > max_odds:
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
        if expectation > epsilon_real:
            return 1. + expectation
        else:
            return 0
    return bet_func_helper


def totals_bet_func(epsilon, bet_totals=True):
    def bet_func_helper(price, odds, spread_prob_win, spread_prob_loss, prediction, bet_row, ml_bet_player,
                        ml_bet_opp, ml_opp_odds):
        if not bet_totals:
            return 0
        alpha_odds = bet_row['overall_odds_avg']
        prediction = prediction * alpha + (1.0 - alpha) * alpha_odds
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

        alpha_odds = bet_row['overall_odds_avg']
        prediction = prediction * alpha + (1.0 - alpha) * alpha_odds
        prediction = spread_prob_win * prediction + spread_prob_loss * (1.0-prediction)

        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)

        if odds < 0.425 or odds > 0.525:
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
    rf = lambda: RandomForestClassifier(n_estimators=300)
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
        #print('With betting model: ', name)
        model_predictions = []
        all_predictions.append(model_predictions)
        seed = int(np.random.randint(0, high=1000000, size=1)) * 2
        if name == 'Random Forest':
            model = _model()
            model.fit(X_train, y_train)
            prob_pos = predict_proba(model, X_test)
            model_predictions.append(prob_pos)
        else:
            for i in range(150):
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

    # parameters
    train_params = [
        [0.33, 0.33, [0., 0.01, 0.025, 0.05, 0.1, 0.20]],
        [0.1, 0.8, [0., 0.01, 0.025, 0.05, 0.1, 0.20]],
        [0.8, 0.1, [0., 0.01, 0.025, 0.05, 0.1, 0.20]],
        [0.1, 0.1, [0., 0.01, 0.025, 0.05, 0.1, 0.20]],
    ]

    test_idx = 0

    if train:
        params = train_params
    else:
        test_params = [
            [train_params[test_idx][0], train_params[test_idx][1], [train_params[test_idx][2][0]]]
        ]
        params = test_params
        # print("Test params: ", params)

    for bayes_model_percent, logit_percent, epsilons in params:
        for epsilon in epsilons:
            if train:
                variance = 0.0001
                bayes_model_percent = bayes_model_percent + float(np.random.randn(1) * variance)
                epsilon = epsilon + float(np.random.randn(1) * variance)
            print('Avg Model ->  Bayes Percentage:', bayes_model_percent, ' Logit Percentage:', logit_percent, ' Epsilon:', epsilon, ' Alpha:', alpha)
            rf_model_percent = 1.0 - logit_percent - bayes_model_percent
           # logit_percent = 1.0 - bayes_model_percent
            total = logit_percent * len(all_predictions[0]) + bayes_model_percent * len(all_predictions[1]) + rf_model_percent * len(all_predictions[2])
            avg_predictions = np.vstack([np.vstack(all_predictions[0]) * logit_percent,
                                         np.vstack(all_predictions[1]) * bayes_model_percent,
                                         np.vstack(all_predictions[2]) * rf_model_percent
                                        ]).sum(0) / total
            predictions.append(avg_predictions)
            if prediction_function is not None:
                prediction_function(avg_predictions, epsilon)
    return predictions


def decision_func(epsilon, bet_ml=True, bet_spread=True, bet_totals=True,
                  bet_on_challengers=False, bet_on_pros=True, bet_on_itf=False,
                  bet_on_clay = False, bet_first_round=True, min_payout=0.92):
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
        ml_payout = ml_bet_option.payout
        spread_payout = spread_bet_option.payout
        totals_payout = totals_bet_option.payout

        #if bet_row['first_round'] > 0.5 or \
        if (bet_row['grand_slam'] > 0.5 and (bet_row['round_num'] < 1 or bet_row['round_num']>5)) or \
                (bet_row['grand_slam'] < 0.5 and (bet_row['round_num'] < 1 or bet_row['round_num']>5)) or \
                (not bet_on_challengers and bet_row['challenger'] > 0.5) or \
                (not bet_on_pros and bet_row['challenger'] < 0.5 and bet_row['itf'] < 0.5) or \
                (not bet_on_itf and bet_row['itf'] > 0.5) or \
                bet_row['opp_prev_year_prior_encounters'] < 3 or \
                (not bet_on_clay and bet_row['clay'] > 0.5) or \
                (not bet_first_round and bet_row['first_row']>0.5) or \
                bet_row['prev_year_prior_encounters'] < 3:
            return {
                'ml_bet1': 0,
                'ml_bet2': 0,
                'spread_bet1': 0,
                'spread_bet2': 0,
                'over_bet': 0,
                'under_bet': 0
            }

        priors_to_use = priors_spread

        spread_prob_win1 = spread_prob(bet_row['player_id'], bet_row['opponent_id'], bet_row['tournament'], bet_row['start_date'],
                                       spread_bet_option.spread1 - spread_cushion, bet_row['grand_slam'] > 0.5, priors_to_use,
                                       bet_row['court_surface'], win=True)
        spread_prob_win2 = spread_prob(bet_row['opponent_id'], bet_row['player_id'], bet_row['tournament'], bet_row['start_date'],
                                       spread_bet_option.spread2 - spread_cushion, bet_row['grand_slam'] > 0.5, priors_to_use,
                                       bet_row['court_surface'], win=True)
        spread_prob_loss1 = spread_prob(bet_row['player_id'], bet_row['opponent_id'], bet_row['tournament'], bet_row['start_date'],
                                        spread_bet_option.spread1 - spread_cushion, bet_row['grand_slam'] > 0.5, priors_to_use,
                                        bet_row['court_surface'], win=False)
        spread_prob_loss2 = spread_prob(bet_row['opponent_id'], bet_row['player_id'], bet_row['tournament'], bet_row['start_date'],
                                        spread_bet_option.spread2 - spread_cushion, bet_row['grand_slam'] > 0.5, priors_to_use,
                                        bet_row['court_surface'], win=False)

        if bet_totals:
            if totals_type_by_betting_site[bet_row['book_name']] == 'Game':
                totals_prob_under_win = total_games_prob(bet_row['player_id'], bet_row['tournament'], bet_row['start_date'],
                                                     totals_bet_option.under, bet_row['grand_slam'] > 0.5, priors_game_totals,
                                                     bet_row['court_surface'], under=True, win=True)
                totals_prob_over_win = total_games_prob(bet_row['opponent_id'], bet_row['tournament'], bet_row['start_date'],
                                                    totals_bet_option.over, bet_row['grand_slam'] > 0.5, priors_game_totals,
                                                    bet_row['court_surface'], under=False, win=True)
                totals_prob_under_loss = total_games_prob(bet_row['player_id'], bet_row['tournament'], bet_row['start_date'],
                                                     totals_bet_option.under, bet_row['grand_slam'] > 0.5, priors_game_totals,
                                                     bet_row['court_surface'], under=True, win=False)
                totals_prob_over_loss = total_games_prob(bet_row['opponent_id'], bet_row['tournament'], bet_row['start_date'],
                                                    totals_bet_option.over, bet_row['grand_slam'] > 0.5, priors_game_totals,
                                                    bet_row['court_surface'], under=False, win=False)
            else:
                totals_prob_under_win = total_sets_prob(bet_row['player_id'], bet_row['tournament'], bet_row['start_date'],
                                                    totals_bet_option.under, bet_row['grand_slam'] > 0.5, priors_set_totals,
                                                    bet_row['court_surface'], win=True, under=True)
                totals_prob_over_win = total_sets_prob(bet_row['opponent_id'], bet_row['tournament'], bet_row['start_date'],
                                                   totals_bet_option.over, bet_row['grand_slam'] > 0.5, priors_set_totals,
                                                   bet_row['court_surface'], win=True, under=False)
                totals_prob_under_loss = total_sets_prob(bet_row['player_id'], bet_row['tournament'], bet_row['start_date'],
                                                    totals_bet_option.under, bet_row['grand_slam'] > 0.5, priors_set_totals,
                                                    bet_row['court_surface'], win=False, under=True)
                totals_prob_over_loss = total_sets_prob(bet_row['opponent_id'], bet_row['tournament'], bet_row['start_date'],
                                                   totals_bet_option.over, bet_row['grand_slam'] > 0.5, priors_set_totals,
                                                   bet_row['court_surface'], win=False, under=False)
        else:
            totals_prob_under_win = 0
            totals_prob_over_win = 0
            totals_prob_under_loss = 0
            totals_prob_over_loss = 0

        if ml_payout < min_payout:
            ml_bet1 = 0
            ml_bet2 = 0
        else:
            ml_bet1 = ml_func(ml_bet_option.max_price1, ml_bet_option.best_odds1, prediction, bet_row)
            ml_bet2 = ml_func(ml_bet_option.max_price2, ml_bet_option.best_odds2, 1.0 - prediction, bet_row)

        if spread_payout < min_payout:
            spread_bet1 = 0
            spread_bet2 = 0
        else:
            spread_bet1 = spread_func(spread_bet_option.max_price1, spread_bet_option.best_odds1, spread_prob_win1, spread_prob_loss1,
                                      prediction, bet_row, ml_bet1, ml_bet2, ml_bet_option.best_odds2)
            spread_bet2 = spread_func(spread_bet_option.max_price2, spread_bet_option.best_odds2, spread_prob_win2, spread_prob_loss2,
                                      1.0 - prediction, bet_row, ml_bet2, ml_bet1, ml_bet_option.best_odds1)

        if totals_payout < min_payout:
            over_bet = 0
            under_bet = 0
        else:
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


def prediction_func(bet_ml=True, bet_spread=True, bet_totals=True,
                  bet_on_challengers=False, bet_on_pros=True, bet_on_itf=False,
                  bet_on_clay = False, bet_first_round=True, min_payout=0.92):
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
                                                                  bet_totals=bet_totals, bet_on_challengers=bet_on_challengers,
                                                                  bet_on_pros=bet_on_pros, bet_on_itf=bet_on_itf,
                                                                  bet_on_clay=bet_on_clay, bet_first_round=bet_first_round, min_payout=min_payout),
                                                    test_data,
                                                    'max_price', 'price', 'totals_price', 1, sampling=0,
                                                    shuffle=True, verbose=False)
        if num_bets is None:
            num_bets = 0
        if num_bets > 0:
            return_per_bet = to_percentage(test_return / (num_bets * 100))
        else:
            return_per_bet = 0
        print('Final test return:', test_return, ' Return per bet:', return_per_bet, ' Num bets:', num_bets, ' Avg Error:',
              to_percentage(avg_error),
              ' Test years:', num_test_years, ' Year:', test_year)
        print('---------------------------------------------------------')
    return prediction_func_helper


start_year = '2012-01-01'

model_names = {
    'All': 'Logistic',
    '25': 'Logistic',
    '100': 'Logistic',
    '500': 'Logistic',
    '1000': 'Logistic',
    '2000': 'Logistic',
    'Clay': 'Logistic',
    'Grass': 'Logistic',
    'Hard': 'Logistic',
    'FirstRound': 'Logistic',
    'OtherRound': 'Logistic'
}

models = {}
for name in model_names:
    models[name] = load_outcome_model(model_names[name]+name)

if __name__ == '__main__':
    num_tests = 1
    bet_spread = True
    bet_ml = True
    bet_totals = False
    bet_on_challengers = False
    bet_on_pros = True
    bet_on_itf = False
    bet_on_clay = False
    bet_first_round = True
    min_payout = 0.92
    if bet_on_itf:
        masters = 0.24
    elif bet_on_challengers:
        masters = 99
    else:
        masters = 101
    for i in range(num_tests):
        print("TEST: ", i)
        num_test_years = 0
        for test_year in [datetime.date.today(), datetime.date(2018, 6, 1), datetime.date(2018, 1, 1), datetime.date(2017, 1, 1)]:
            for num_test_months in [3, 6, 12]:
                graph = False
                all_predictions = []
                data, test_data = load_data(start_year=start_year, num_test_years=num_test_years,
                                            test_year=test_year, models=models, spread_models=None, masters_min=masters, num_test_months=num_test_months)
                avg_predictions = predict(data, test_data, prediction_function=prediction_func(min_payout=min_payout, bet_on_pros=bet_on_pros, bet_on_itf=bet_on_itf, bet_on_challengers=bet_on_challengers, bet_on_clay=bet_on_clay, bet_first_round=bet_first_round, bet_ml=bet_ml, bet_spread=bet_spread, bet_totals=bet_totals), graph=False, train=True)

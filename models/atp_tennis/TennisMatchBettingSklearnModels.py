from models.atp_tennis.TennisMatchOutcomeLogit import test_model, to_percentage
from models.atp_tennis.TennisMatchRNN import load_nn, predict_nn
import models.atp_tennis.TennisMatchRNN as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
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

totals_type_by_betting_site = {  # describes the totals type for each betting site
    'bovada': 'Set',
    'dimes': 'Game',
    'betonline': 'Game',
    #'OddsPortal': 'Game'
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
    'predictions_nn',
    'predictions_spread',
]

for attr in betting_only_attributes:
    betting_input_attributes.append(attr)

betting_only_attributes.append('ml_odds_avg')
betting_only_attributes.append('overall_odds_avg')


all_attributes = list(betting_input_attributes)
meta_attributes = ['player_id', 'opponent_id', 'tournament', 'year', 'ml_odds_avg']
for meta in meta_attributes:
    if meta not in all_attributes:
        all_attributes.append(meta)

for attr in nn.additional_attributes:
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
        select m.start_date,m.tournament,m.team1 as team1,m.team2 as team2,
        s.bovada_price1 as price1,
        s.bovada_price2 as price2,
        s.bovada_spread1 as spread1,
        s.bovada_spread2 as spread2,
        s.bovada_odds1 as odds1,
        s.bovada_odds2 as odds2,
        s.dimes_price1 as price1,
        s.dimes_price2 as price2,
        s.dimes_spread1 as spread1,
        s.dimes_spread2 as spread2,
        s.dimes_odds1 as odds1,
        s.dimes_odds2 as odds2,
        s.betonline_price1 as price1,
        s.betonline_price2 as price2,
        s.betonline_spread1 as spread1,
        s.betonline_spread2 as spread2,
        s.betonline_odds1 as odds1,
        s.betonline_odds2 as odds2,
        coalesce(m.betting_date,s.betting_date) as betting_date,
        m.best_odds1 as ml_odds1,
        m.best_odds2 as ml_odds2,   
        (m.avg_odds1+(1.0-m.avg_odds2))/2.0 as ml_odds_avg,
        m.best_price1 as max_price1,
        m.best_price2 as max_price2,
        100 as totals_price1,
        100 as totals_price2,
        100 as over,
        100 as under,
        ml_overall_avg.avg_odds as overall_odds_avg  
        from atp_tennis_betting_link_common as m 
        left outer join atp_tennis_betting_link_spread_common as s
        on ((m.team1,m.team2,m.tournament,m.start_date)=(s.team1,s.team2,s.tournament,s.start_date))
        left outer join atp_matches_money_line_average as ml_overall_avg
            on ((m.team1,m.team2,m.start_date,m.tournament)=(ml_overall_avg.player_id,ml_overall_avg.opponent_id,ml_overall_avg.start_date,ml_overall_avg.tournament))        
        where m.start_date<='{{YEAR}}'::date
    union all
        select m.start_date,m.tournament,m.team2 as team1,m.team1 as team2,
        s.bovada_price2 as bovada_price1,
        s.bovada_price1 as bovada_price2,
        s.bovada_spread2 as bovada_spread1,
        s.bovada_spread1 as bovada_spread2,
        s.bovada_odds2 as bovada_odds1,
        s.bovada_odds1 as bovada_odds2,
        s.dimes_price2 as dimes_price1,
        s.dimes_price1 as dimes_price2,
        s.dimes_spread2 as dimes_spread1,
        s.dimes_spread1 as dimes_spread2,
        s.dimes_odds2 as dimes_odds1,
        s.dimes_odds1 as dimes_odds2,
        s.betonline_price2 as betonline_price1,
        s.betonline_price1 as betonline_price2,
        s.betonline_spread2 as betonline_spread1,
        s.betonline_spread1 as betonline_spread2,
        s.betonline_odds2 as betonline_odds1,
        s.betonline_odds1 as betonline_odds2,     
        coalesce(m.betting_date,s.betting_date) as betting_date,
        m.best_odds2 as ml_odds1,   
        m.best_odds1 as ml_odds2,
        (m.avg_odds2+(1.0-m.avg_odds1))/2.0 as ml_odds_avg,
        m.best_price2 as max_price1,
        m.best_price1 as max_price2,
        100 as totals_price1,
        100 as totals_price2,
        100 as over,
        100 as under,
        1.0 - ml_overall_avg.avg_odds as overall_odds_avg  
        from atp_tennis_betting_link_common as m 
        left outer join atp_tennis_betting_link_spread_common as s
        on ((m.team1,m.team2,m.tournament,m.start_date)=(s.team1,s.team2,s.tournament,s.start_date))
        left outer join atp_matches_money_line_average as ml_overall_avg
            on ((m.team1,m.team2,m.start_date,m.tournament)=(ml_overall_avg.player_id,ml_overall_avg.opponent_id,ml_overall_avg.start_date,ml_overall_avg.tournament))        
        where m.start_date<='{{YEAR}}'::date      
    '''.replace('{{YEAR}}', str(test_year.strftime('%Y-%m-%d'))), conn)
    return betting_data


def extract_beat_spread_binary(spreads, spread_actuals):
    res = []
    ties = 0
    for i in range(len(spreads)):
        spread = spreads[i]
        spread_actual = spread_actuals[i]
        if np.isnan(spread_actual) or np.isnan(spread):
            r = 0.
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


def load_outcome_predictions_and_actuals(attributes, test_tournament=None, test_year=datetime.date.today(), start_year='2005-01-01', masters_min=101):
    data, _ = tennis_model.get_all_data(attributes, tournament=test_tournament, test_season=test_year.strftime('%Y-%m-%d'), num_test_years=0, start_year=start_year, masters_min=masters_min, num_test_months=0)
    start_year = datetime.datetime.strptime(start_year, '%Y-%m-%d').date()
    data = nn.merge_data(data, start_year, test_year)
    return data


def load_data(start_year, test_year, test_tournament=None, masters_min=101, attributes=list(tennis_model.all_attributes)):
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
    data = load_outcome_predictions_and_actuals(attributes, test_tournament=test_tournament, start_year=start_year, test_year=test_year, masters_min=masters_min)

    betting_data = load_betting_data(betting_sites, test_year=test_year)

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
    def mask_func(x):
        if np.isnan(x):
            return 0.
        else:
            return 1.

    #data = data.assign(spread_mask=pd.Series([mask_func(x) for x in data['spread1'].iloc[:]]).values)
    #data.sort_values(by=['betting_date'], inplace=True, ascending=True, kind='mergesort')
    #test_data.sort_values(by=['betting_date'], inplace=True, ascending=True, kind='mergesort')
    #data.reset_index(drop=True, inplace=True)
    #test_data.reset_index(drop=True, inplace=True)
    return data


alpha = 1.0
spread_cushion = 0.0
dont_bet_against_spread = {}


def bet_func(epsilon, bet_ml=True, useRatio=True):
    def bet_func_helper(price, odds, prediction, bet_row):
        if not bet_ml:
            return 0

        alpha_odds = bet_row['overall_odds_avg']
        prediction = prediction * alpha + (1.0 - alpha) * alpha_odds
        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)

        min_odds = 0.20
        max_odds = 0.60

        if odds < min_odds or odds > max_odds:
            return 0

        if useRatio:
            expectation = (prediction / odds) - 1.0
        else:
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


def totals_bet_func(epsilon, bet_totals=True, useRatio=True):
    def bet_func_helper(price, odds, spread_prob_win, spread_prob_loss, prediction, bet_row):
        if not bet_totals:
            return 0
        alpha_odds = bet_row['overall_odds_avg']
        prediction = prediction * alpha + (1.0 - alpha) * alpha_odds
        prediction = spread_prob_win * prediction + spread_prob_loss * (1.0 - prediction)

        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)

        if odds < 0.40 or odds > 0.5:
            return 0

        if useRatio:
            expectation = (prediction / odds) - 1.0
        else:
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


def spread_bet_func(epsilon, bet_spread=True, useRatio=True):
    def bet_func_helper(price, odds, spread_prob_win):
        if not bet_spread:
            return 0

        prediction = spread_prob_win

        if 0 > prediction or prediction > 1:
            print('Invalid prediction: ', prediction)
            exit(1)

        if odds < 0.45 or odds > 0.525:
            return 0

        if useRatio:
            expectation = (prediction / odds) - 1.0
        else:
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


def predict(predictions, prediction_function, train=False):
    # parameters
    epsilons = [0., 0.01, 0.025, 0.05, 0.1, 0.20]
    if not train:
        epsilons = [0.]

    if prediction_function is not None:
        for epsilon in epsilons:
            prediction_function(predictions, epsilon)
    return predictions


def spread_prob(predictions_spread, spread, win=True):
    spread_pos = -int(spread) + 1
    prob = predictions_spread[18 + spread_pos:].sum()
    if not win:
        prob = 1.0 - prob
    return prob


def decision_func(epsilon, bet_ml=True, bet_spread=True, bet_totals=True,
                  bet_on_challengers=False, bet_on_pros=True, bet_on_itf=False,
                  bet_on_clay = False, bet_first_round=True, min_payout=0.92):
    ml_func = bet_func(epsilon, bet_ml=bet_ml)
    spread_func = spread_bet_func(epsilon, bet_spread=bet_spread)
    totals_func = totals_bet_func(epsilon, bet_totals=bet_totals)

    def check_player_for_spread(bet, opponent):
        if opponent in dont_bet_against_spread:
            return 0.
        return bet

    def decision_func_helper(ml_bet_option, spread_bet_option, totals_bet_option, bet_row, prediction):
        ml_payout = ml_bet_option.payout
        spread_payout = spread_bet_option.payout
        totals_payout = totals_bet_option.payout

        if (bet_row['grand_slam'] > 0.5 and (bet_row['round_num'] < 1 or bet_row['round_num']>5)) or \
                (bet_row['grand_slam'] < 0.5 and (bet_row['round_num'] < 1 or bet_row['round_num']>5)) or \
                (not bet_on_challengers and bet_row['challenger'] > 0.5) or \
                (not bet_on_pros and bet_row['challenger'] < 0.5 and bet_row['itf'] < 0.5) or \
                (not bet_on_itf and bet_row['itf'] > 0.5) or \
                bet_row['opp_prev_year_prior_encounters'] < 3 or \
                (not bet_on_clay and bet_row['clay'] > 0.5) or \
                (not bet_first_round and bet_row['first_round']>0.5) or \
                (bet_row['is_qualifier'] > 0.5) or \
                bet_row['prev_year_prior_encounters'] < 3:
            return {
                'ml_bet1': 0,
                'ml_bet2': 0,
                'spread_bet1': 0,
                'spread_bet2': 0,
                'over_bet': 0,
                'under_bet': 0
            }

        spread_prob_win1 = spread_prob(np.array(bet_row['predictions_spread']).flatten(),
                                       spread_bet_option.spread1 - spread_cushion, win=True)
        spread_prob_loss1 = spread_prob(np.array(bet_row['predictions_spread']).flatten(),
                                        spread_bet_option.spread1 - spread_cushion, win=False)

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

        if not bet_ml or ml_payout < min_payout:
            ml_bet1 = 0
        else:
            ml_bet1 = ml_func(ml_bet_option.max_price1, ml_bet_option.best_odds1, prediction, bet_row)

        if not bet_spread or spread_payout < min_payout:
            spread_bet1 = 0
        else:
            spread_bet1 = spread_func(spread_bet_option.max_price1, spread_bet_option.best_odds1, spread_prob_win1)

        if not bet_totals or totals_payout < min_payout:
            over_bet = 0
            under_bet = 0
        else:
            over_bet = totals_func(totals_bet_option.max_price1, totals_bet_option.best_odds1, totals_prob_over_win, totals_prob_over_loss,
                                   prediction, bet_row)
            under_bet = totals_func(totals_bet_option.max_price2, totals_bet_option.best_odds2, totals_prob_under_win, totals_prob_under_loss,
                                    1.0 - prediction, bet_row)

        # check whether it's an amazing player in a grand slam
        #   basically, we don't want to bet against them...
        if bet_row['grand_slam'] > 0.5:
            spread_bet1 = check_player_for_spread(spread_bet1, bet_row['opponent_id'])

        return {
            'ml_bet1': ml_bet1,
            'ml_bet2': 0,
            'spread_bet1': spread_bet1,
            'spread_bet2': 0,
            'over_bet': over_bet,
            'under_bet': under_bet
        }
    return decision_func_helper


def flip(m, axis):
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]


def prediction_func(test_data, bet_ml=True, bet_spread=True, bet_totals=True,
                  bet_on_challengers=False, bet_on_pros=True, bet_on_itf=False,
                  bet_on_clay = False, bet_first_round=True, min_payout=0.92):
    def prediction_func_helper(avg_predictions, epsilon):
        _, _, _, avg_error = tennis_model.score_predictions(avg_predictions, test_data['spread'].iloc[:])

        def game_or_set_totals_func(i):
            if totals_type_by_betting_site[test_data['book_name'].iloc[i]] == 'Game':
                return test_data['totals'].iloc[i]
            else:
                return test_data['num_sets'].iloc[i]

        total_test_return = 0.0
        total_num_bets = 0

        if bet_ml or bet_totals:
            test_return, num_bets = simulate_money_line(lambda j: avg_predictions[j],
                                                        lambda j: test_data['y'].iloc[j],
                                                        lambda j: test_data['spread'].iloc[j],
                                                        lambda j: game_or_set_totals_func(j),
                                                        decision_func(epsilon, bet_ml=bet_ml, bet_spread=False,
                                                                      bet_totals=bet_totals, bet_on_challengers=bet_on_challengers,
                                                                      bet_on_pros=bet_on_pros, bet_on_itf=bet_on_itf,
                                                                      bet_on_clay=bet_on_clay, bet_first_round=bet_first_round, min_payout=min_payout),
                                                        test_data,
                                                        'max_price', 'price', 'totals_price', 1, sampling=0,
                                                        shuffle=True, verbose=False)
            total_test_return += test_return
            total_num_bets += num_bets

        if bet_spread:
            for book in betting_sites:
                test_return, num_bets = simulate_money_line(lambda j: avg_predictions[j],
                                                            lambda j: test_data['y'].iloc[j],
                                                            lambda j: test_data['spread'].iloc[j],
                                                            lambda j: game_or_set_totals_func(j),
                                                            decision_func(epsilon, bet_ml=False, bet_spread=bet_spread,
                                                                          bet_totals=False,
                                                                          bet_on_challengers=bet_on_challengers,
                                                                          bet_on_pros=bet_on_pros,
                                                                          bet_on_itf=bet_on_itf,
                                                                          bet_on_clay=bet_on_clay,
                                                                          bet_first_round=bet_first_round,
                                                                          min_payout=min_payout),
                                                            test_data,
                                                            'max_price', book+'_price', 'totals_price', 1, sampling=0,
                                                            shuffle=True, verbose=False)

                total_test_return += test_return
                total_num_bets += num_bets
        if total_num_bets is None:
            total_num_bets = 0
        if total_num_bets > 0:
            return_per_bet = to_percentage(total_test_return / (total_num_bets * 100))
        else:
            return_per_bet = 0
        print('Final test return:', total_test_return, ' Return per bet:', return_per_bet, ' Num bets:', total_num_bets, ' Avg Error:',
              to_percentage(avg_error),
              ' Test years:', num_test_years, ' Test Months:', num_test_months, ' Year:', test_year)
        print('---------------------------------------------------------')
    return prediction_func_helper


start_year = '2015-01-01'

if __name__ == '__main__':
    model = load_nn()
    bet_spread = True
    bet_ml = True
    bet_totals = False
    bet_on_challengers = False
    bet_on_pros = True
    bet_on_itf = False
    bet_on_clay = True
    bet_first_round = True
    min_payout = 0.92
    if bet_on_itf:
        masters = 24
    elif bet_on_challengers:
        masters = 99
    else:
        masters = 101

    data = load_data(start_year=start_year, test_year=datetime.date.today(), masters_min=masters)
    if model is not None:
        print('Making predictions...')
        y_nn = predict_nn(model, data)
        print('Done.')
        data = data.assign(
            predictions_nn=pd.Series([y_nn[0][i] for i in range(data.shape[0])]).values)
        data = data.assign(
            predictions_spread=pd.Series([y_nn[1][i] for i in range(data.shape[0])]).values)
    all_labels = list(data.columns.values)
    print("ALL ATTRS: ", all_labels)
    print("TEST")
    num_test_years = 0
    for test_year in [datetime.date(2018, 6, 1), datetime.date(2018, 1, 1), datetime.date(2017, 6, 1)]:
        for num_test_months in [12, 18, 24]:
            graph = False
            all_predictions = []
            date = tennis_model.get_date_from(test_year.strftime('%Y-%m-%d'), num_test_years, num_test_months)
            print("Using date: ", date)
            test_data = data[data.start_date >= datetime.datetime.strptime(date, '%Y-%m-%d').date()]
            test_data = test_data[test_data.start_date < test_year]
            predictions = np.array(test_data['predictions_nn'])
            predict(predictions, prediction_function=prediction_func(test_data,
                min_payout=min_payout, bet_on_pros=bet_on_pros, bet_on_itf=bet_on_itf,
                bet_on_challengers=bet_on_challengers, bet_on_clay=bet_on_clay, bet_first_round=bet_first_round,
                bet_ml=bet_ml, bet_spread=bet_spread, bet_totals=bet_totals
            ), train=True)

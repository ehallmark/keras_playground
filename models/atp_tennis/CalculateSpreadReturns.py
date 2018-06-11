import keras as k
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from numpy.random import shuffle
import models.atp_tennis.TennisMatchOutcomeNN as tennis_model
from models.atp_tennis.TennisMatchOutcomeNN import test_model,to_percentage

model = k.models.load_model('tennis_match_keras_nn_v3.h5')
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
print(model.summary())

test_year = 2018  # IMPORTANT!!
all_data = tennis_model.get_all_data(test_year)
test_meta_data = all_data[2]
test_data = all_data[1]
test_labels = test_data[1]
avg_error = test_model(model, test_data[0], test_labels)
print('Average error: ', to_percentage(avg_error))

predictions = model.predict(test_data[0])
predictions[0] = predictions[0].flatten()
predictions[1] = predictions[1].flatten()

betting_sites = ['Bovada','5Dimes','BetOnline']
conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
betting_data = pd.read_sql('''
    select year,tournament,team1,team2,
    price1,
    price2,
    spread1,
    spread2
    from atp_tennis_betting_link_spread 
    where year={{YEAR}} and book_name in ({{BOOK_NAMES}})
'''.replace('{{YEAR}}', str(test_year)).replace('{{BOOK_NAMES}}', '\''+'\',\''.join(betting_sites)+'\''), conn)

price_str = 'price'

prev_best = -1000000
prev_worst = 1000000
avg_best = 0
count = 0
best_parameters = None
worst_parameters = None
parameters = {}
np.random.seed(1)


def betting_decision(victory_prediction, spread_prediction, odds, spread, underdog, parameters={}):
    if underdog:
        if victory_prediction > 0.5 + parameters['betting_epsilon']:
            return True
        elif victory_prediction > odds - spread/20. + parameters['betting_epsilon'] and spread - spread_prediction < parameters['spread_epsilon']:
            # check spread and prediction
            return True
        else:
            return False
    else:
        if victory_prediction > odds - spread/20. + parameters['betting_epsilon']: # formula for spread
            return True
        elif victory_prediction > odds - parameters['betting_epsilon'] and spread - spread_prediction > parameters['spread_epsilon']:
            return True
        else:
            return False
num_trials = 50
for trial in range(num_trials):
    print('Trial: ',trial)
    parameters['max_loss_percent'] = 0.05
    parameters['betting_epsilon'] = 0.15 + (np.random.rand(1)*0.02 - 0.01)
    parameters['spread_epsilon'] = 3.0 + (np.random.rand(1) * 3.0 - 1.5)
    parameters['max_price_plus'] = 200
    parameters['max_price_minus'] = -180
    return_total = 0.0
    num_bets = 0
    num_wins = 0
    num_losses = 0
    amount_invested = 0
    amount_won = 0
    amount_lost = 0
    num_wins1 = 0
    num_wins2 = 0
    num_losses1 = 0
    num_losses2 = 0
    betting_minimum = 10.0
    initial_capital = 1000.0
    num_ties = 0
    available_capital = initial_capital
    indices = list(range(test_meta_data.shape[0]))
    shuffle(indices)
    print('Indices: ', indices)
    for i in indices:
        row = test_meta_data.iloc[i]
        prediction = predictions[0][i]
        spread_prediction = predictions[1][i]
        # prediction = np.random.rand(1)  # test on random predictions
        bet_rows = betting_data[
            (betting_data.year == row.year) &
            (betting_data.team1 == row.player_id) &
            (betting_data.team2 == row.opponent_id) &
            (betting_data.tournament == row.tournament)
        ]
        if bet_rows.shape[0] >= 1:
            for r in range(bet_rows.shape[0]):
                bet_row = bet_rows.iloc[r]
                # make betting decision
                max_price1 = np.array(bet_row[price_str+'1']).flatten()[0]
                max_price2 = np.array(bet_row[price_str+'2']).flatten()[0]
                # calculate odds ratio
                '''
                Implied probability	=	( - ( 'minus' moneyline odds ) ) / ( - ( 'minus' moneyline odds ) ) + 100
                Implied probability	=	100 / ( 'plus' moneyline odds + 100 )
                '''
                spread1 = bet_row['spread1']
                spread2 = bet_row['spread2']
                is_under1 = spread1 < 0
                is_under2 = spread2 < 0
                is_price_under1 = max_price1 < 0
                is_price_under2 = max_price2 < 0
                if max_price1 > 0:
                    best_odds1 = 100.0 / (100.0 + max_price1)
                else:
                    best_odds1 = -1.0 * (max_price1 / (-1.0 * max_price1 + 100.0))
                if max_price2 > 0:
                    best_odds2 = 100.0 / (100.0 + max_price2)
                else:
                    best_odds2 = -1.0 * (max_price2 / (-1.0 * max_price2 + 100.0))

                if best_odds1 < 0.0 or best_odds1 > 1.0:
                    raise ArithmeticError('Best odds1: ' + str(best_odds1))
                if best_odds2 < 0.0 or best_odds2 > 1.0:
                    raise ArithmeticError('Best odds2: '+str(best_odds2))
                #print('Found best odds 1: ', best_odds1)
                #print('Found best odds 2: ', best_odds2)
                #print('Found prediction: ', prediction)
                return_game = 0.0
                actual_result = test_labels[0][i]
                actual_spread = test_labels[1][i]
                player1_win = None
                player2_win = None
                beat_spread1 = None
                beat_spread2 = None
                if actual_result > 0.5:  # player 1 wins match
                    player1_win = True
                    player2_win = False
                    if is_under1:  # was favorite
                        beat_spread1 = abs(spread1) < actual_spread
                    else:
                        beat_spread1 = True  # by default
                    if is_under2:
                        beat_spread2 = False
                    else:
                        beat_spread2 = abs(spread2) > actual_spread

                else:
                    player1_win = False
                    player2_win = True
                    if is_under1:
                        beat_spread1 = False
                    else:
                        beat_spread1 = abs(spread1) > -actual_spread
                    if is_under2:  # opponent was favorite
                        beat_spread2 = abs(spread2) < -actual_spread
                    else:
                        beat_spread2 = True

                #print("Spreads: ", spread1, spread2, actual_spread, spread_prediction)
                #print("Victories: ", player1_win, player2_win, "Beat spreads: ", beat_spread1, beat_spread2)
                bet1 = spread1 + spread_prediction >= parameters['spread_epsilon'] or (not is_under1 and prediction>0.5+parameters['betting_epsilon'])
                if bet1 and parameters['max_price_minus'] < max_price1 < parameters['max_price_plus'] and best_odds1 < prediction - parameters['betting_epsilon']:
                    confidence = betting_minimum # (prediction - best_odds1) * betting_minimum
                    if is_price_under1:
                        capital_requirement = -max_price1 * confidence
                    else:
                        capital_requirement = 100.0 * confidence
                    #print('Initial capital requirement1: ', capital_requirement)
                    capital_requirement_avail = max(betting_minimum, min(parameters['max_loss_percent']*available_capital, capital_requirement))
                    capital_ratio = capital_requirement_avail/capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if capital_requirement <= available_capital:
                        #print('Confidence: ', confidence)
                        #print('Max price1: ', max_price1, 'Max price2: ',max_price2)
                        #print('Capital Req: ', capital_requirement)
                        #print('Make BET! Advantage', prediction - best_odds1)
                        #print('Capital ratio: ', capital_ratio)
                        amount_invested += capital_requirement
                        if beat_spread1 is None:  # tie
                            ret = 0
                            num_ties += 1
                         #   print('TIE!!!!')
                        elif beat_spread1:  # WON BET
                         #   print('WON!!')
                            if is_price_under1:
                                ret = 100.0 * confidence
                            else:
                                ret = max_price1 * confidence
                            if ret < 0:
                                raise ArithmeticError("win 1 should be positive")
                            num_wins = num_wins + 1
                            num_wins1 += 1
                            amount_won += abs(ret)
                        else:
                         #   print('LOST!!')
                            if is_price_under1:
                                ret = max_price1 * confidence
                            else:
                                ret = - 100.0 * confidence
                            if ret > 0:
                                raise ArithmeticError("Loss 1 should be positive")
                            num_losses = num_losses + 1
                            amount_lost += abs(ret)
                            num_losses1 += 1

                        return_game += ret
                        available_capital += ret
                        num_bets += 1
                        #print('Ret 1: ', ret)
                bet2 = spread2 - spread_prediction >= parameters['spread_epsilon'] or (not is_under2 and prediction<0.5-parameters['betting_epsilon'])
                if bet2 and parameters['max_price_minus'] < max_price2 < parameters['max_price_plus'] and best_odds2 < (1.0 - prediction) - parameters['betting_epsilon']:
                    confidence = betting_minimum # (1.0 - prediction - best_odds2) * betting_minimum
                    if is_price_under2:
                        capital_requirement = -max_price2 * confidence
                    else:
                        capital_requirement = 100.0 * confidence

                    #print('Initial capital requirement1: ', capital_requirement)
                    capital_requirement_avail = max(betting_minimum, min(parameters['max_loss_percent'] * available_capital, capital_requirement))
                    capital_ratio = capital_requirement_avail / capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if capital_requirement <= available_capital:
                        #print('Confidence: ', confidence)
                        #print('Max price1: ', max_price1, 'Max price2: ', max_price2)
                        #print('Capital Req: ', capital_requirement)
                        #print('Make BET on OPPONENT! Advantage', (1.0-prediction)-best_odds2)
                        #print('Capital ratio: ', capital_ratio)
                        amount_invested += capital_requirement
                        if beat_spread2 is None:
                            ret = 0  # tie
                            num_ties += 1
                         #   print('TIE!!!!!!!')
                        elif beat_spread2:  # WON BET
                          #  print('WON!!')
                            if is_price_under2:
                                ret = 100.0 * confidence
                            else:
                                ret = max_price2 * confidence
                            if ret < 0:
                                raise ArithmeticError("win 2 should be positive")
                            num_wins += 1
                            num_wins2 += 1
                            amount_won += abs(ret)
                        else:  # LOST BET :(
                          #  print('LOST!!')
                            if is_price_under2:
                                ret = max_price2 * confidence
                            else:
                                ret = - 100.0 * confidence
                            if ret > 0:
                                raise ArithmeticError("loss 2 should be negative")
                            num_losses2 += 1
                            num_losses += 1
                            amount_lost += abs(ret)
                        return_game += ret
                        num_bets += 1
                        available_capital += ret
                        #print('Ret 2: ', ret)
                #print('Return for the match: ', return_game)
                return_total = return_total + return_game
                #print('Num bets: ', num_bets)
                #print('Capital: ', available_capital)
                #print('Return Total: ', return_total)

    print("Parameters: ", parameters)
    #print('Initial Capital: ', initial_capital)
    print('Final Capital: ', available_capital)
    print('Num bets: ', num_bets)
    print('Total Return: ', return_total)
    #print('Amount invested: ', amount_invested)
    #print('Amount won: ', amount_won)
    #print('Amount lost: ', amount_lost)
    #print('Average Return Per Amount Invested: ', return_total / amount_invested)
    print('Overall Return For The Year: ', return_total / initial_capital)
    print('Num correct: ', num_wins)
    print('Num wrong: ', num_losses)
    #print('Num correct1: ', num_wins1)
    #print('Num wrong1: ', num_losses1)
    #print('Num correct2: ', num_wins2)
    #print('Num wrong2: ', num_losses2)
    #print('Num ties: ', num_ties)
    avg_best += return_total
    if return_total > prev_best:
        prev_best = return_total
        best_parameters = parameters.copy()
    if return_total < prev_worst:
        prev_worst = return_total
        worst_parameters = parameters.copy()

print('Best return: ', prev_best)
print('Worst return', prev_worst)
print('Avg return', avg_best/num_trials)
print('Best Parameters: ', best_parameters)
print('Worst parameters: ', worst_parameters)

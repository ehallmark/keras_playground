import numpy as np
import statsmodels.formula.api as smf
from random import shuffle


def simulate_money_line(predictor_func, actual_label_func, parameter_update_func, betting_epsilon_func, test_meta_data, betting_data, parameters,
                        price_str='price', num_trials=50):
    worst_parameters = None
    best_parameters = None
    avg_best = 0.0
    prev_best = -1000000.0
    prev_worst = 1000000.0
    regression_data = {}
    parameter_update_func(parameters)
    for key in parameters:
        regression_data[key] = []
    regression_data['return_total'] = []
    for trial in range(num_trials):
        #print('Trial: ',trial)
        parameter_update_func(parameters)
        for key in parameters:  # add params to regression map
            regression_data[key].append(parameters[key])
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
        available_capital = initial_capital
        indices = list(range(test_meta_data.shape[0]))
        shuffle(indices)
        for i in indices:
            row = test_meta_data.iloc[i]
            prediction = predictor_func(i)
            # prediction = np.random.rand(1)  # test on random predictions
            bet_row = betting_data[
                (betting_data.year == row.year) &
                (betting_data.team1 == row.player_id) &
                (betting_data.team2 == row.opponent_id) &
                (betting_data.tournament == row.tournament)
            ]
            if bet_row.shape[0] == 1:
                # make betting decision
                max_price1 = np.array(bet_row[price_str+'1']).flatten()[0]
                max_price2 = np.array(bet_row[price_str+'2']).flatten()[0]
                # calculate odds ratio
                '''
                Implied probability	=	( - ( 'minus' moneyline odds ) ) / ( - ( 'minus' moneyline odds ) ) + 100
                Implied probability	=	100 / ( 'plus' moneyline odds + 100 )
                '''
                is_under1 = max_price1 < 0
                is_under2 = max_price2 < 0
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

                return_game = 0.0
                actual_result = actual_label_func(i)

                price_diff = abs(abs(max_price1) - abs(max_price2))
                if 'max_price_diff' in parameters and price_diff > parameters['max_price_diff']:
                    continue
                if parameters['max_price_minus'] < max_price1 < parameters['max_price_plus'] and best_odds1 < prediction - betting_epsilon_func(max_price1):
                    confidence = (prediction - best_odds1) * betting_minimum
                    if is_under1:
                        capital_requirement = -max_price1 * confidence
                    else:
                        capital_requirement = 100.0 * confidence
                    capital_requirement_avail = max(betting_minimum, min(parameters['max_loss_percent']*available_capital, capital_requirement))
                    capital_ratio = capital_requirement_avail/capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if capital_requirement <= available_capital:
                        amount_invested += capital_requirement
                        if actual_result < 0.5:  # LOST BET :(
                            if is_under1:
                                ret = max_price1 * confidence
                            else:
                                ret = - 100.0 * confidence
                            if ret > 0:
                                raise ArithmeticError("Loss 1 should be positive")
                            num_losses = num_losses + 1
                            amount_lost += abs(ret)
                            num_losses1 += 1
                        else:  # WON BET
                            if is_under1:
                                ret = 100.0 * confidence
                            else:
                                ret = max_price1 * confidence
                            if ret < 0:
                                raise ArithmeticError("win 1 should be positive")
                            num_wins = num_wins + 1
                            num_wins1 += 1
                            amount_won += abs(ret)
                        return_game += ret
                        available_capital += ret
                        num_bets += 1
                if parameters['max_price_minus'] < max_price2 < parameters['max_price_plus'] and best_odds2 < (1.0 - prediction) - betting_epsilon_func(max_price2):
                    confidence = (1.0 - prediction - best_odds2) * betting_minimum
                    if is_under2:
                        capital_requirement = -max_price2 * confidence
                    else:
                        capital_requirement = 100.0 * confidence

                    capital_requirement_avail = max(betting_minimum, min(parameters['max_loss_percent'] * available_capital, capital_requirement))
                    capital_ratio = capital_requirement_avail / capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if capital_requirement <= available_capital:
                        amount_invested += capital_requirement
                        if actual_result < 0.5:  # WON BET
                            if is_under2:
                                ret = 100.0 * confidence
                            else:
                                ret = max_price2 * confidence
                            if ret < 0:
                                raise ArithmeticError("win 2 should be positive")
                            num_wins += 1
                            num_wins2 += 1
                            amount_won += abs(ret)
                        else:  # LOST BET :(
                            if is_under2:
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
                return_total = return_total + return_game
    #    print("Parameters: ", parameters)
    #    print('Initial Capital: ', initial_capital)
    #    print('Final Capital: ', available_capital)
    #    print('Num bets: ', num_bets)
    #    print('Total Return: ', return_total)
    #    print('Average Return Per Amount Invested: ', return_total / amount_invested)
    #    print('Overall Return For The Year: ', return_total / initial_capital)
    #    print('Num correct: ', num_wins)
    #    print('Num wrong: ', num_losses)
        regression_data['return_total'].append(return_total)
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
    #print('Best Parameters: ', best_parameters)
    #print('Worst parameters: ', worst_parameters)

    # model to predict the total score (h_pts + a_pts)
    #results = smf.OLS(np.array(regression_data['return_total']), np.array([regression_data['betting_epsilon1'],regression_data['betting_epsilon2']]).transpose()).fit()
    #print(results.summary())


def simulate_spread(predictor_func, spread_predictor_func, actual_label_func, actual_spread_func, parameter_update_func, betting_decision_func, test_meta_data, betting_data, parameters,
                        price_str='price', num_trials=50):
    worst_parameters = None
    best_parameters = None
    avg_best = 0.0
    prev_best = -1000000.0
    prev_worst = 1000000.0
    regression_data = {}
    parameter_update_func(parameters)
    for key in parameters:
        regression_data[key] = []
    regression_data['return_total'] = []
    for trial in range(num_trials):
        print('Trial: ', trial)
        parameter_update_func(parameters)
        for key in parameters:  # add params to regression map
            regression_data[key].append(parameters[key])
        max_price_diff = 200.
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
            prediction = predictor_func(i)
            spread_prediction = spread_predictor_func(i)
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
                    max_price1 = np.array(bet_row[price_str + '1']).flatten()[0]
                    max_price2 = np.array(bet_row[price_str + '2']).flatten()[0]
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
                    price_diff = abs(abs(max_price1) - abs(max_price2))
                    if max_price_diff is not None and price_diff > max_price_diff:
                        continue
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
                        raise ArithmeticError('Best odds2: ' + str(best_odds2))
                    return_game = 0.0
                    actual_result = actual_label_func(i)
                    actual_spread = actual_spread_func(i)
                    if actual_result > 0.5:  # player 1 wins match
                        if is_under1:  # was favorite
                            if abs(abs(spread1) - actual_spread) < 0.000001:
                                beat_spread1 = None
                            else:
                                beat_spread1 = abs(spread1) < actual_spread
                        else:
                            beat_spread1 = True  # by default
                        if is_under2:
                            beat_spread2 = False
                        else:
                            if abs(abs(spread2) - actual_spread) < 0.000001:
                                beat_spread2 = None
                            else:
                                beat_spread2 = abs(spread2) > actual_spread

                    else:
                        if is_under1:
                            beat_spread1 = False
                        else:
                            if abs(abs(spread1) - -actual_spread) < 0.000001:
                                beat_spread1 = None
                            else:
                                beat_spread1 = abs(spread1) > -actual_spread
                        if is_under2:  # opponent was favorite
                            if abs(abs(spread2) - -actual_spread) < 0.000001:
                                beat_spread2 = None
                            else:
                                beat_spread2 = abs(spread2) < -actual_spread
                        else:
                            beat_spread2 = True

                    bet1 = betting_decision_func(prediction, spread_prediction, best_odds1, spread1, not is_under1,
                                            parameters)
                    if bet1 and parameters['max_price_minus'] < max_price1 < parameters['max_price_plus']:
                        confidence = betting_minimum  # (prediction - best_odds1) * betting_minimum
                        if is_price_under1:
                            capital_requirement = -max_price1 * confidence
                        else:
                            capital_requirement = 100.0 * confidence
                        # print('Initial capital requirement1: ', capital_requirement)
                        capital_requirement_avail = max(betting_minimum,
                                                        min(parameters['max_loss_percent'] * available_capital,
                                                            capital_requirement))
                        capital_ratio = capital_requirement_avail / capital_requirement
                        capital_requirement *= capital_ratio
                        confidence *= capital_ratio
                        if capital_requirement <= available_capital:
                            amount_invested += capital_requirement
                            if beat_spread1 is None:  # tie
                                ret = 0
                                num_ties += 1
                            elif beat_spread1:  # WON BET
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
                    bet2 = betting_decision_func(1.0 - prediction, -spread_prediction, best_odds2, spread2, not is_under2,
                                            parameters)
                    if bet2 and parameters['max_price_minus'] < max_price2 < parameters['max_price_plus']:
                        confidence = betting_minimum  # (1.0 - prediction - best_odds2) * betting_minimum
                        if is_price_under2:
                            capital_requirement = -max_price2 * confidence
                        else:
                            capital_requirement = 100.0 * confidence

                        capital_requirement_avail = max(betting_minimum,
                                                        min(parameters['max_loss_percent'] * available_capital,
                                                            capital_requirement))
                        capital_ratio = capital_requirement_avail / capital_requirement
                        capital_requirement *= capital_ratio
                        confidence *= capital_ratio
                        if capital_requirement <= available_capital:
                            amount_invested += capital_requirement
                            if beat_spread2 is None:
                                ret = 0  # tie
                                num_ties += 1
                            #   print('TIE!!!!!!!')
                            elif beat_spread2:  # WON BET
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
                    return_total = return_total + return_game

        #print("Parameters: ", parameters)
        # print('Initial Capital: ', initial_capital)
        #print('Final Capital: ', available_capital)
        print('Num bets: ', num_bets)
        #print('Total Return: ', return_total)
        # print('Amount invested: ', amount_invested)
        # print('Amount won: ', amount_won)
        # print('Amount lost: ', amount_lost)
        # print('Average Return Per Amount Invested: ', return_total / amount_invested)
        #print('Overall Return For The Year: ', return_total / initial_capital)
        #print('Num correct: ', num_wins)
        #print('Num wrong: ', num_losses)
        #print('Num ties: ', num_ties)
        # print('Num correct1: ', num_wins1)
        # print('Num wrong1: ', num_losses1)
        # print('Num correct2: ', num_wins2)
        # print('Num wrong2: ', num_losses2)
        # print('Num ties: ', num_ties)
        avg_best += return_total
        regression_data['return_total'].append(return_total)
        if return_total > prev_best:
            prev_best = return_total
            best_parameters = parameters.copy()
        if return_total < prev_worst:
            prev_worst = return_total
            worst_parameters = parameters.copy()

    print('Best return: ', prev_best)
    print('Worst return', prev_worst)
    print('Avg return', avg_best / num_trials)
    #print('Best Parameters: ', best_parameters)
    #print('Worst parameters: ', worst_parameters)

    # model to predict the total score (h_pts + a_pts)
    results = smf.OLS(np.array(regression_data['return_total']), np.array(
        [regression_data['betting_epsilon1'], regression_data['betting_epsilon2'],
         regression_data['spread_epsilon']]).transpose()).fit()
    print(results.summary())

    return avg_best

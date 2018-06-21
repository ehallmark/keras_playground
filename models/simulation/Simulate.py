import numpy as np
import statsmodels.formula.api as smf
from random import shuffle
import pandas as pd


def simulate_money_line(predictor_func, actual_label_func, bet_func, test_meta_data,
                        price_str='price', num_trials=50, sampling=0, verbose=False, shuffle=False):

    avg_best = 0.0
    num_bets_total = 0
    for trial in range(num_trials):
        #print('Trial: ',trial)
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
        betting_minimum = 5.0
        betting_maximum = 100.0
        max_loss_percent = 0.05
        initial_capital = 10000.0
        available_capital = initial_capital
        indices = list(range(test_meta_data.shape[0]))
        if shuffle:
            np.random.shuffle(indices)
        if sampling > 0:
            indices = indices[0: round(sampling*len(indices))]
        for i in indices:
            row = test_meta_data.iloc[i]
            prediction = predictor_func(i)
            #prediction = float(np.random.rand(1))  # test on random predictions
            bet_row = row
            if bet_row[price_str+str(1)] != np.nan:
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

                bet1 = bet_func(max_price1, best_odds1, prediction)
                if bet1 > 0:
                    confidence = betting_minimum * bet1
                    if is_under1:
                        capital_requirement = -max_price1 * confidence
                    else:
                        capital_requirement = 100.0 * confidence
                    capital_requirement_avail = max(betting_minimum, min(min(max_loss_percent*available_capital, betting_maximum), capital_requirement))
                    capital_ratio = capital_requirement_avail/capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if capital_requirement <= available_capital:
                        amount_invested += capital_requirement
                        if verbose:
                            print('Invested:', capital_requirement, ' Odds:', best_odds1)
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
                bet2 = bet_func(max_price2, best_odds2, 1.0-prediction)
                if bet2 > 0:
                    confidence = betting_minimum * bet2
                    if is_under2:
                        capital_requirement = -max_price2 * confidence
                    else:
                        capital_requirement = 100.0 * confidence

                    capital_requirement_avail = max(betting_minimum, min(min(max_loss_percent*available_capital, betting_maximum), capital_requirement))
                    capital_ratio = capital_requirement_avail / capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if capital_requirement <= available_capital:
                        amount_invested += capital_requirement
                        if verbose:
                            print('Invested:', capital_requirement,' Odds:', best_odds2)

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
                if return_game != 0 and verbose:
                    print('Return game:', return_game, ' Total: ', return_total, ' Wins:',num_wins, ' Losses:', num_losses)
        if verbose:
            print('Initial Capital: ', initial_capital)
            print('Final Capital: ', available_capital)
            print('Num bets: ', num_bets)
            print('Total Return: ', return_total)
            print('Average Return Per Amount Invested: ', return_total / max(1,amount_invested))
            print('Overall Return For The Year: ', return_total / initial_capital)
            print('Num correct: ', num_wins)
            print('Num wrong: ', num_losses)
        avg_best += return_total
        num_bets_total += num_bets

    avg_best /= num_trials
    num_bets_avg = num_bets_total / num_trials
    #print('Best return: ', prev_best)
    #print('Worst return', prev_worst)
    #print('Avg return', avg_best)
    #print('Best Parameters: ', best_parameters)
    #print('Worst parameters: ', worst_parameters)

    # model to predict the total score (h_pts + a_pts)
    #results = smf.OLS(np.array(regression_data['return_total']), np.array([regression_data['betting_epsilon1'],regression_data['betting_epsilon2']]).transpose()).fit()
    #print(results.summary())
    return avg_best, num_bets_avg


def simulate_spread(predictor_func, actual_spread_func,
                    betting_decision_func, test_data,
                    price_str='price', num_trials=50, sampling=0, verbose=False, shuffle=True):
    worst_parameters = None
    best_parameters = None
    avg_best = 0.0
    total_num_bets = 0
    prev_best = -1000000.0
    prev_worst = 1000000.0
    for trial in range(num_trials):
        #print('Trial: ', trial)
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
        betting_minimum = 5.0
        betting_maximum = 100.0
        max_loss_percent = 0.05
        initial_capital = 10000.0
        num_ties = 0
        available_capital = initial_capital
        indices = list(range(test_data.shape[0]))
        if shuffle:
            np.random.shuffle(indices)
        if sampling > 0:
            indices = indices[0: round(sampling*len(indices))]
        for i in indices:
            prediction = predictor_func(i)
            # prediction = np.random.rand(1)  # test on random predictions
            bet_row = test_data.iloc[i]
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

            is_price_under1 = max_price1 < 0
            is_price_under2 = max_price2 < 0
            price_diff = abs(abs(max_price1) - abs(max_price2))

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
            actual_spread = actual_spread_func(i)
            if spread1 > 0:
                if spread1 == -actual_spread:
                    beat_spread1 = None
                else:
                    beat_spread1 = spread1 > -actual_spread
            else:
                if -spread1 == actual_spread:
                    beat_spread1 = None
                else:
                    beat_spread1 = -spread1 < actual_spread

            if spread2 > 0:
                if spread2 == actual_spread:
                    beat_spread2 = None
                else:
                    beat_spread2 = spread2 > actual_spread
            else:
                if -spread2 == -actual_spread:
                    beat_spread2 = None
                else:
                    beat_spread2 = -spread2 < -actual_spread

            bet1 = betting_decision_func(max_price1, best_odds1, spread1, prediction)
            if bet1 > 0:
                confidence = betting_minimum * bet1
                if is_price_under1:
                    capital_requirement = -max_price1 * confidence
                else:
                    capital_requirement = 100.0 * confidence
                # print('Initial capital requirement1: ', capital_requirement)
                capital_requirement_avail = max(betting_minimum,
                                                min(min(max_loss_percent * available_capital, betting_maximum),
                                                    capital_requirement))
                capital_ratio = capital_requirement_avail / capital_requirement
                capital_requirement *= capital_ratio
                confidence *= capital_ratio
                if capital_requirement <= available_capital:
                    amount_invested += capital_requirement
                    if verbose:
                        print('Invested: ', capital_requirement)
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
                    if verbose:
                        if ret < 0:
                            print(' MINUS: ', ret)
                        else:
                            print(' PLUS: ', ret)
            bet2 = betting_decision_func(max_price2, best_odds2, spread2, 1.0-prediction)
            if bet2 > 0:
                confidence = betting_minimum * bet2
                if is_price_under2:
                    capital_requirement = -max_price2 * confidence
                else:
                    capital_requirement = 100.0 * confidence

                capital_requirement_avail = max(betting_minimum,
                                                min(min(max_loss_percent * available_capital, betting_maximum),
                                                    capital_requirement))
                capital_ratio = capital_requirement_avail / capital_requirement
                capital_requirement *= capital_ratio
                confidence *= capital_ratio
                if capital_requirement <= available_capital:
                    amount_invested += capital_requirement
                    if verbose:
                        print('Invested: ', capital_requirement)
                    if beat_spread2 is None:
                        ret = 0  # tie
                        num_ties += 1
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
                    if verbose:
                        if ret < 0:
                            print(' MINUS: ', ret)
                        else:
                            print(' PLUS: ', ret)

            return_total = return_total + return_game
            if return_game != 0 and verbose:
                print('return game:', return_game, ' Total return:', return_total, ' Wins:', num_wins, ' Losses:', num_losses)
        if verbose:
            #print("Parameters: ", parameters)
            print('Initial Capital: ', initial_capital)
            print('Final Capital: ', available_capital)
            print('Num bets: ', num_bets)
            print('Total Return: ', return_total)
            print('Amount invested: ', amount_invested)
            print('Amount won: ', amount_won)
            print('Amount lost: ', amount_lost)
            print('Average Return Per Amount Invested: ', return_total / amount_invested)
            print('Overall Return For The Year: ', return_total / initial_capital)
            print('Num correct: ', num_wins)
            print('Num wrong: ', num_losses)
            print('Num ties: ', num_ties)
            print('Num correct1: ', num_wins1)
            print('Num wrong1: ', num_losses1)
            print('Num correct2: ', num_wins2)
            print('Num wrong2: ', num_losses2)
            print('Num ties: ', num_ties)
        avg_best += return_total
        total_num_bets += num_bets

    avg_best /= num_trials
    #print('Best return: ', prev_best)
    #print('Worst return', prev_worst)
    #print('Avg return', avg_best)
    #print('Best Parameters: ', best_parameters)
    #print('Worst parameters: ', worst_parameters)

    # model to predict the total return
    #results = smf.OLS(np.array(regression_data['return_total']), np.array(
    #    [regression_data['betting_epsilon1'], regression_data['betting_epsilon2'],
    #     regression_data['spread_epsilon']]).transpose()).fit()
    #print(results.summary())
    avg_num_bets = total_num_bets / num_trials
    return avg_best, avg_num_bets

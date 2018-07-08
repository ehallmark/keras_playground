import numpy as np
import statsmodels.formula.api as smf
from random import shuffle
import pandas as pd
import math


class BetOption:
    def __init__(self, bet_row, price_str):
        self.max_price1 = np.array(bet_row[price_str + '1']).flatten()[0]
        self.max_price2 = np.array(bet_row[price_str + '2']).flatten()[0]
        # calculate odds ratio
        '''
        Implied probability	=	( - ( 'minus' moneyline odds ) ) / ( - ( 'minus' moneyline odds ) ) + 100
        Implied probability	=	100 / ( 'plus' moneyline odds + 100 )
        '''
        self.spread1 = bet_row['spread1']
        self.spread2 = bet_row['spread2']
        self.over = bet_row['over']
        self.under = bet_row['under']
        self.is_under1 = self.max_price1 < 0
        self.is_under2 = self.max_price2 < 0
        if self.max_price1 > 0:
            self.best_odds1 = 100.0 / (100.0 + self.max_price1)
        else:
            self.best_odds1 = -1.0 * (self.max_price1 / (-1.0 * self.max_price1 + 100.0))
        if self.max_price2 > 0:
            self.best_odds2 = 100.0 / (100.0 + self.max_price2)
        else:
            self.best_odds2 = -1.0 * (self.max_price2 / (-1.0 * self.max_price2 + 100.0))

        if self.best_odds1 < 0.0 or self.best_odds1 > 1.0:
            raise ArithmeticError('Best odds1: ' + str(self.best_odds1))
        if self.best_odds2 < 0.0 or self.best_odds2 > 1.0:
            raise ArithmeticError('Best odds2: ' + str(self.best_odds2))


def simulate_money_line(predictor_func, actual_label_func, actual_spread_func, decision_func, test_meta_data,
                        price_str='max_price', spread_price_str='price', totals_price_str='totals_price', num_trials=50, sampling=0, verbose=False,
                        shuffle=False, initial_capital=10000, after_bet_function=None):

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
        won_both = 0
        lost_both = 0
        won_spread_lost_ml = 0
        lost_spread_won_ml = 0
        betting_minimum = 5.0
        betting_maximum = 100.0
        max_loss_percent = 0.05
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
                return_game = 0.0

                ml_bet_option = BetOption(bet_row, price_str)
                spread_bet_option = BetOption(bet_row, spread_price_str)
                totals_bet_option = BetOption(bet_row, totals_price_str)
                bet_decision = decision_func(ml_bet_option, spread_bet_option, totals_bet_option, bet_row, prediction)
                ml_bet1 = bet_decision['ml_bet1']
                ml_bet2 = bet_decision['ml_bet2']
                spread_bet1 = bet_decision['spread_bet1']
                spread_bet2 = bet_decision['spread_bet2']
                under_bet = bet_decision['under_bet']
                over_bet = bet_decision['over_bet']

                # spread data
                spread_price_is_under1 = spread_bet_option.is_under1
                spread_price_is_under2 = spread_bet_option.is_under2

                # total data
                totals_price_is_under1 = totals_bet_option.is_under1
                totals_price_is_under2 = totals_bet_option.is_under2

                ml_is_under1 = ml_bet_option.is_under1
                ml_is_under2 = ml_bet_option.is_under2
                ml_price1 = ml_bet_option.max_price1
                ml_price2 = ml_bet_option.max_price2
                spread_price1 = spread_bet_option.max_price1
                spread_price2 = spread_bet_option.max_price2
                spread1 = spread_bet_option.spread1
                spread2 = spread_bet_option.spread2

                # money line result
                actual_result = actual_label_func(i)
                actual_spread = actual_spread_func(i)

                won_ml = None
                won_spread = None
                won_totals = None

                if ml_bet1 > 0:
                    confidence = betting_minimum * ml_bet1
                    if ml_is_under1:
                        capital_requirement = -ml_price1 * confidence
                    else:
                        capital_requirement = 100.0 * confidence
                    capital_requirement_avail = max(betting_minimum, min(min(max_loss_percent*available_capital, betting_maximum), capital_requirement))
                    capital_ratio = capital_requirement_avail/capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if after_bet_function is not None:
                        after_bet_function(ml_bet1, bet_row['player_id'], bet_row['opponent_id'], 'ML',
                                           ml_price1, capital_requirement, bet_row['betting_date'],
                                           bet_row['book_name'])

                    if capital_requirement <= available_capital:
                        amount_invested += capital_requirement
                        if actual_result < 0.5:  # LOST BET :(
                            if ml_is_under1:
                                ret = ml_price1 * confidence
                            else:
                                ret = - 100.0 * confidence
                            if ret > 0:
                                raise ArithmeticError("Loss 1 should be positive")
                            num_losses = num_losses + 1
                            num_losses1 += 1
                            amount_lost += abs(ret)
                            won_ml = False
                        else:  # WON BET
                            if ml_is_under1:
                                ret = 100.0 * confidence
                            else:
                                ret = ml_price1 * confidence
                            if ret < 0:
                                raise ArithmeticError("win 1 should be positive")
                            num_wins = num_wins + 1
                            num_wins1 += 1
                            won_ml = True
                            amount_won += abs(ret)
                        return_game += ret
                        available_capital += ret
                        num_bets += 1
                if ml_bet2 > 0:
                    confidence = betting_minimum * ml_bet2
                    if ml_is_under2:
                        capital_requirement = -ml_price2 * confidence
                    else:
                        capital_requirement = 100.0 * confidence

                    capital_requirement_avail = max(betting_minimum, min(min(max_loss_percent*available_capital, betting_maximum), capital_requirement))
                    capital_ratio = capital_requirement_avail / capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if after_bet_function is not None:
                        after_bet_function(ml_bet2, bet_row['opponent_id'], bet_row['player_id'], 'ML',
                                           ml_price2, capital_requirement, bet_row['betting_date'],
                                           bet_row['book_name'])
                    if capital_requirement <= available_capital:
                        amount_invested += capital_requirement
                        if actual_result < 0.5:  # WON BET
                            if ml_is_under2:
                                ret = 100.0 * confidence
                            else:
                                ret = ml_price2 * confidence
                            if ret < 0:
                                raise ArithmeticError("win 2 should be positive")
                            num_wins += 1
                            num_wins1 += 1
                            won_ml = True
                            amount_won += abs(ret)
                        else:  # LOST BET :(
                            if ml_is_under2:
                                ret = ml_price2 * confidence
                            else:
                                ret = - 100.0 * confidence
                            if ret > 0:
                                raise ArithmeticError("loss 2 should be negative")
                            num_losses += 1
                            num_losses1 += 1
                            won_ml = False
                            amount_lost += abs(ret)
                        return_game += ret
                        num_bets += 1
                        available_capital += ret

                if spread1 > 0:
                    if math.isclose(spread1, -actual_spread):
                        beat_spread1 = None
                    else:
                        beat_spread1 = spread1 > -actual_spread
                else:
                    if math.isclose(-spread1, actual_spread):
                        beat_spread1 = None
                    else:
                        beat_spread1 = -spread1 < actual_spread

                if spread2 > 0:
                    if math.isclose(spread2, actual_spread):
                        beat_spread2 = None
                    else:
                        beat_spread2 = spread2 > actual_spread
                else:
                    if math.isclose(-spread2, -actual_spread):
                        beat_spread2 = None
                    else:
                        beat_spread2 = -spread2 < -actual_spread
                if spread_bet1 > 0:
                    confidence = betting_minimum * spread_bet1
                    if spread_price_is_under1:
                        capital_requirement = -spread_price1 * confidence
                    else:
                        capital_requirement = 100.0 * confidence
                    # print('Initial capital requirement1: ', capital_requirement)
                    capital_requirement_avail = max(betting_minimum,
                                                    min(min(max_loss_percent * available_capital, betting_maximum),
                                                        capital_requirement))
                    capital_ratio = capital_requirement_avail / capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if after_bet_function is not None:
                        after_bet_function(spread_bet1, bet_row['player_id'], bet_row['opponent_id'], spread1,
                                           spread_price1, capital_requirement, bet_row['betting_date'], bet_row['book_name'])
                    if capital_requirement <= available_capital:
                        amount_invested += capital_requirement
                        if verbose:
                            print('Invested: ', capital_requirement)
                        if beat_spread1 is None:  # tie
                            ret = 0
                        elif beat_spread1:  # WON BET
                            if spread_price_is_under1:
                                ret = 100.0 * confidence
                            else:
                                ret = spread_price1 * confidence
                            if ret < 0:
                                raise ArithmeticError("win 1 should be positive")
                            num_wins = num_wins + 1
                            num_wins2 += 1
                            won_spread = True
                            amount_won += abs(ret)
                        else:
                            if spread_price_is_under1:
                                ret = spread_price1 * confidence
                            else:
                                ret = - 100.0 * confidence
                            if ret > 0:
                                raise ArithmeticError("Loss 1 should be positive")
                            num_losses = num_losses + 1
                            num_losses2 += 1
                            won_spread = False
                            amount_lost += abs(ret)

                        return_game += ret
                        available_capital += ret
                        num_bets += 1
                        if verbose:
                            if ret < 0:
                                print(' MINUS: ', ret)
                            else:
                                print(' PLUS: ', ret)
                if spread_bet2 > 0:
                    confidence = betting_minimum * spread_bet2
                    if spread_price_is_under2:
                        capital_requirement = -spread_price2 * confidence
                    else:
                        capital_requirement = 100.0 * confidence

                    capital_requirement_avail = max(betting_minimum,
                                                    min(min(max_loss_percent * available_capital, betting_maximum),
                                                        capital_requirement))
                    capital_ratio = capital_requirement_avail / capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if after_bet_function is not None:
                        after_bet_function(spread_bet2, bet_row['opponent_id'], bet_row['player_id'], spread2,
                                           spread_price2, capital_requirement, bet_row['betting_date'], bet_row['book_name'])
                    if capital_requirement <= available_capital:
                        amount_invested += capital_requirement
                        if verbose:
                            print('Invested: ', capital_requirement)
                        if beat_spread2 is None:
                            ret = 0  # tie
                        elif beat_spread2:  # WON BET
                            if spread_price_is_under2:
                                ret = 100.0 * confidence
                            else:
                                ret = spread_price2 * confidence
                            if ret < 0:
                                raise ArithmeticError("win 2 should be positive")
                            num_wins += 1
                            num_wins2 += 1
                            won_spread = True
                            amount_won += abs(ret)
                        else:  # LOST BET :(
                            if spread_price_is_under2:
                                ret = spread_price2 * confidence
                            else:
                                ret = - 100.0 * confidence
                            if ret > 0:
                                raise ArithmeticError("loss 2 should be negative")
                            num_losses += 1
                            num_losses2 += 1
                            won_spread = False
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
                    print('Return game:', return_game, ' Total: ', return_total, ' Wins:', num_wins, ' Losses:', num_losses)
                if won_spread is not None and won_ml is not None:
                    if won_spread and won_ml:
                        won_both += 1
                    elif won_spread and not won_ml:
                        won_spread_lost_ml += 1
                    elif not won_spread and won_ml:
                        lost_spread_won_ml += 1
                    else:
                        lost_both += 1

                if over_bet > 0:
                    confidence = betting_minimum * ml_bet1
                    if ml_is_under1:
                        capital_requirement = -ml_price1 * confidence
                    else:
                        capital_requirement = 100.0 * confidence
                    capital_requirement_avail = max(betting_minimum, min(min(max_loss_percent*available_capital, betting_maximum), capital_requirement))
                    capital_ratio = capital_requirement_avail/capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if after_bet_function is not None:
                        after_bet_function(ml_bet1, bet_row['player_id'], bet_row['opponent_id'], 'ML',
                                           ml_price1, capital_requirement, bet_row['betting_date'],
                                           bet_row['book_name'])

                    if capital_requirement <= available_capital:
                        amount_invested += capital_requirement
                        if actual_result < 0.5:  # LOST BET :(
                            if ml_is_under1:
                                ret = ml_price1 * confidence
                            else:
                                ret = - 100.0 * confidence
                            if ret > 0:
                                raise ArithmeticError("Loss 1 should be positive")
                            num_losses = num_losses + 1
                            num_losses1 += 1
                            amount_lost += abs(ret)
                            won_ml = False
                        else:  # WON BET
                            if ml_is_under1:
                                ret = 100.0 * confidence
                            else:
                                ret = ml_price1 * confidence
                            if ret < 0:
                                raise ArithmeticError("win 1 should be positive")
                            num_wins = num_wins + 1
                            num_wins1 += 1
                            won_ml = True
                            amount_won += abs(ret)
                        return_game += ret
                        available_capital += ret
                        num_bets += 1
                if under_bet > 0:
                    confidence = betting_minimum * ml_bet2
                    if ml_is_under2:
                        capital_requirement = -ml_price2 * confidence
                    else:
                        capital_requirement = 100.0 * confidence

                    capital_requirement_avail = max(betting_minimum, min(min(max_loss_percent*available_capital, betting_maximum), capital_requirement))
                    capital_ratio = capital_requirement_avail / capital_requirement
                    capital_requirement *= capital_ratio
                    confidence *= capital_ratio
                    if after_bet_function is not None:
                        after_bet_function(ml_bet2, bet_row['opponent_id'], bet_row['player_id'], 'ML',
                                           ml_price2, capital_requirement, bet_row['betting_date'],
                                           bet_row['book_name'])
                    if capital_requirement <= available_capital:
                        amount_invested += capital_requirement
                        if actual_result < 0.5:  # WON BET
                            if ml_is_under2:
                                ret = 100.0 * confidence
                            else:
                                ret = ml_price2 * confidence
                            if ret < 0:
                                raise ArithmeticError("win 2 should be positive")
                            num_wins += 1
                            num_wins1 += 1
                            won_ml = True
                            amount_won += abs(ret)
                        else:  # LOST BET :(
                            if ml_is_under2:
                                ret = ml_price2 * confidence
                            else:
                                ret = - 100.0 * confidence
                            if ret > 0:
                                raise ArithmeticError("loss 2 should be negative")
                            num_losses += 1
                            num_losses1 += 1
                            won_ml = False
                            amount_lost += abs(ret)
                        return_game += ret
                        num_bets += 1
                        available_capital += ret

        print('ML wins:', num_wins1, ' ML losses:', num_losses1)
        print('Spread wins:', num_wins2, ' Spread losses:', num_losses2)
        print('won_both:', won_both, ' lost_both:', lost_both, ' won_spread_lost_ml:', won_spread_lost_ml, ' lost_spread_won_ml:', lost_spread_won_ml)
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


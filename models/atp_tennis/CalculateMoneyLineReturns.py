import keras as k
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import create_engine
from random import shuffle
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

'''
create table atp_tennis_betting_link (
    year int not null,
    book_id integer not null,
    tournament text not null,
    book_name text not null,
    team1 text not null,
    team2 text not null,
    price1 decimal(10,2) not null,
    price2 decimal(10,2) not null,
    betting_date date not null,
    primary key(year,book_id,tournament,team1,team2)
);
'''

print('Test Meta Data Size: ', test_meta_data.shape[0])

betting_sites = ['Bovada','5Dimes','BetOnline']
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
    where year={{YEAR}} and book_name in ({{BOOK_NAMES}})
    group by year,tournament,team1,team2
'''.replace('{{YEAR}}', str(test_year)).replace('{{BOOK_NAMES}}', '\''+'\',\''.join(betting_sites)+'\''), conn)
'''
    Use max_price for the 'best' price,
    Use min_price for the 'worst' price,
    Use avg_price for the 'average' price
'''
price_str = 'max_price'
num_trials = 50
parameters = {}

def betting_epsilon(price):
    return parameters['betting_epsilon']


avg_best = 0
prev_worst = 1000000
prev_best = -1000000
regression_data = {}
regression_data['max_price_diff'] = []
regression_data['betting_epsilon'] = []
regression_data['return_total'] = []
worst_parameters = None
best_parameters = None
for trial in range(num_trials):
    print('Trial: ',trial)
    parameters['max_loss_percent'] = 0.05
    parameters['betting_epsilon'] = 0.20 + (np.random.rand(1)*0.10 - 0.05)
    parameters['max_price_plus'] = 200
    parameters['max_price_minus'] = -180
    parameters['max_price_diff'] = 100.0
    regression_data['betting_epsilon'].append(parameters['betting_epsilon'])
    regression_data['max_price_diff'].append(parameters['max_price_diff'])
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
        prediction = predictions[0][i]
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
            #print('Found best odds 1: ', best_odds1)
            #print('Found best odds 2: ', best_odds2)
            #print('Found prediction: ', prediction)
            return_game = 0.0
            actual_result = test_labels[0][i]

            price_diff = abs(abs(max_price1) - abs(max_price2))
            if price_diff > parameters['max_price_diff']:
                # print('Skipping large price diff: ', price_diff)
                continue
            if parameters['max_price_minus'] < max_price1 < parameters['max_price_plus'] and best_odds1 < prediction - betting_epsilon(max_price1):
                confidence = (prediction - best_odds1) * betting_minimum
                if is_under1:
                    capital_requirement = -max_price1 * confidence
                else:
                    capital_requirement = 100.0 * confidence
               # print('Initial capital requirement1: ', capital_requirement)
                capital_requirement_avail = max(betting_minimum, min(parameters['max_loss_percent']*available_capital, capital_requirement))
                capital_ratio = capital_requirement_avail/capital_requirement
                capital_requirement *= capital_ratio
                confidence *= capital_ratio
                if capital_requirement <= available_capital:
               #     print('Confidence: ', confidence)
               #     print('Max price1: ', max_price1, 'Max price2: ',max_price2)
               #     print('Capital Req: ', capital_requirement)
                #    print('Make BET! Advantage', prediction - best_odds1)
               #     print('Capital ratio: ', capital_ratio)
                    amount_invested += capital_requirement
                    if actual_result < 0.5:  # LOST BET :(
               #         print('LOST!!')
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
               #         print('WON!!')
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
              #      print('Ret 1: ', ret)
            if parameters['max_price_minus'] < max_price2 < parameters['max_price_plus'] and best_odds2 < (1.0 - prediction) - betting_epsilon(max_price2):
                confidence = (1.0 - prediction - best_odds2) * betting_minimum
                if is_under2:
                    capital_requirement = -max_price2 * confidence
                else:
                    capital_requirement = 100.0 * confidence

              #  print('Initial capital requirement1: ', capital_requirement)
                capital_requirement_avail = max(betting_minimum, min(parameters['max_loss_percent'] * available_capital, capital_requirement))
                capital_ratio = capital_requirement_avail / capital_requirement
                capital_requirement *= capital_ratio
                confidence *= capital_ratio
                if capital_requirement <= available_capital:
              #      print('Confidence: ', confidence)
              #      print('Max price1: ', max_price1, 'Max price2: ', max_price2)
              #      print('Capital Req: ', capital_requirement)
              #      print('Make BET on OPPONENT! Advantage', (1.0-prediction)-best_odds2)
              #      print('Capital ratio: ', capital_ratio)
                    amount_invested += capital_requirement
                    if actual_result < 0.5:  # WON BET
              #          print('WON!!')
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
             #           print('LOST!!')
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
             #       print('Ret 2: ', ret)
            #print('Return for the match: ', return_game)
            return_total = return_total + return_game
            #print('Num bets: ', num_bets)
            #print('Capital: ', available_capital)
            #print('Return Total: ', return_total)
    print("Parameters: ", parameters)
    print('Initial Capital: ', initial_capital)
    print('Final Capital: ', available_capital)
    print('Num bets: ', num_bets)
    print('Total Return: ', return_total)
    print('Average Return Per Amount Invested: ', return_total / amount_invested)
    print('Overall Return For The Year: ', return_total / initial_capital)
    print('Num correct: ', num_wins)
    print('Num wrong: ', num_losses)
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
print('Best Parameters: ', best_parameters)
print('Worst parameters: ', worst_parameters)

# model to predict the total score (h_pts + a_pts)
results = smf.OLS(np.array(regression_data['return_total']),np.array([regression_data['betting_epsilon']])).fit()
print(results.summary())

exit(0)

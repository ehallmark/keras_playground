import keras as k
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from random import shuffle
import models.atp_tennis.TennisMatchOutcomeNN as tennis_model
from models.atp_tennis.TennisMatchOutcomeNN import test_model,to_percentage

model = k.models.load_model('tennis_match_keras_nn_v2.h5')
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
print(model.summary())

test_year = 2018  # IMPORTANT!!
all_data = tennis_model.get_all_data(test_year)
test_meta_data = all_data[2]
test_data = all_data[1]
test_labels = test_data[1]
binary_correct, n, binary_percent, avg_error = test_model(model, test_data[0], test_labels)
print('Correctly predicted: ' + str(binary_correct) + ' out of ' + str(n) +
      ' (' + to_percentage(binary_percent) + ')')
print('Average error: ', to_percentage(avg_error))

predictions = model.predict(test_data[0]).flatten()
binary_predictions = (predictions >= 0.5).astype(int)
print('predictions: ', binary_predictions)

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
print('Predictions Size: ', len(binary_predictions))

betting_sites = ['Bovada']
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

betting_epsilon = 0.01
print(betting_data[0:10])
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
initial_capital = 450.0
max_loss_percent = 0.1
available_capital = initial_capital
indices = list(range(test_meta_data.shape[0]))
shuffle(indices)
for i in indices:
    row = test_meta_data.iloc[i]
    prediction = predictions[i]
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
        actual_result = test_labels[i]
        if max_price1 > 0 and best_odds1 < prediction - betting_epsilon:
            confidence = (prediction - best_odds1) * betting_minimum
            capital_requirement = 100.0 * confidence
            print('Initial capital requirement1: ', capital_requirement)
            capital_requirement_avail = max(betting_minimum, min(max_loss_percent*available_capital, capital_requirement))
            capital_ratio = capital_requirement_avail/capital_requirement
            capital_requirement *= capital_ratio
            confidence *= capital_ratio
            if capital_requirement <= available_capital:
                print('Confidence: ', confidence)
                print('Max price1: ', max_price1, 'Max price2: ',max_price2)
                print('Capital Req: ', capital_requirement)
                print('Make BET! Advantage', prediction - best_odds1)
                print('Capital ratio: ', capital_ratio)
                amount_invested += capital_requirement
                if actual_result < 0.5:  # LOST BET :(
                    print('LOST!!')
                    ret = - 100.0 * confidence
                    if ret > 0:
                        raise ArithmeticError("Loss 1 should be positive")
                    num_losses = num_losses + 1
                    amount_lost += abs(ret)
                    num_losses1 += 1
                else:  # WON BET
                    print('WON!!')
                    ret = max_price1 * confidence
                    if ret < 0:
                        raise ArithmeticError("win 1 should be positive")
                    num_wins = num_wins + 1
                    num_wins1 += 1
                    amount_won += abs(ret)
                return_game += ret
                available_capital += ret
                num_bets += 1
                print('Ret 1: ', ret)
        if max_price2 > 0 and best_odds2 < (1.0 - prediction) - betting_epsilon:
            confidence = (1.0 - prediction - best_odds2) * betting_minimum
            capital_requirement = 100.0 * confidence
            print('Initial capital requirement1: ', capital_requirement)
            capital_requirement_avail = max(betting_minimum, min(max_loss_percent * available_capital, capital_requirement))
            capital_ratio = capital_requirement_avail / capital_requirement
            capital_requirement *= capital_ratio
            confidence *= capital_ratio
            if capital_requirement <= available_capital:
                print('Confidence: ', confidence)
                print('Max price1: ', max_price1, 'Max price2: ', max_price2)
                print('Capital Req: ', capital_requirement)
                print('Make BET on OPPONENT! Advantage', (1.0-prediction)-best_odds2)
                print('Capital ratio: ', capital_ratio)
                amount_invested += capital_requirement
                if actual_result < 0.5:  # WON BET
                    print('WON!!')
                    ret = max_price2 * confidence
                    if ret < 0:
                        raise ArithmeticError("win 2 should be positive")
                    num_wins += 1
                    num_wins2 += 1
                    amount_won += abs(ret)
                else:  # LOST BET :(
                    print('LOST!!')
                    ret = - 100.0 * confidence
                    if ret > 0:
                        raise ArithmeticError("loss 2 should be negative")
                    num_losses2 += 1
                    num_losses += 1
                    amount_lost += abs(ret)
                return_game += ret
                num_bets += 1
                available_capital += ret
                print('Ret 2: ', ret)
        print('Return for the match: ', return_game)
        return_total = return_total + return_game
        print('Num bets: ', num_bets)
        print('Capital: ', available_capital)
        print('Return Total: ', return_total)

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
print('Num correct1: ', num_wins1)
print('Num wrong1: ', num_losses1)
print('Num correct2: ', num_wins2)
print('Num wrong2: ', num_losses2)

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
all_data = tennis_model.get_all_data(test_year,test_year,tournament='roland-garros')
test_meta_data = all_data[2]
test_data = all_data[1][0]

print("Meta Data: ", test_meta_data[0:10])
print('Test data: ', test_data[0:10])


predictions = model.predict(test_data).flatten()

# run betting algo

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
betting_minimum = 5.0
initial_capital = 450.0
max_loss_percent = 0.1
available_capital = initial_capital
indices = list(range(test_meta_data.shape[0]))
round = 'Quarter-Finals'
betting_epsilon = 0.01
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
    if bet_row.shape[0] == 1 and 'del-potro' in row['player_id']:
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
        best_odds2 = 1.0 - best_odds2  # reverse odds for loss

        if best_odds1 < 0.0 or best_odds1 > 1.0:
            raise ArithmeticError('Best odds1: ' + str(best_odds1))
        if best_odds2 < 0.0 or best_odds2 > 1.0:
            raise ArithmeticError('Best odds2: '+str(best_odds2))
        #print('Found best odds 1: ', best_odds1)
        #print('Found best odds 2: ', best_odds2)
        #print('Found prediction: ', prediction)
        return_game = 0.0
        if max_price1 > 0 and best_odds1 < prediction - betting_epsilon:
            #print('Make BET! Advantage', prediction-best_odds1)
            confidence = (prediction - best_odds1) * betting_minimum
            capital_requirement = 100.0 * confidence
            capital_requirement_avail = max(betting_minimum, min(max_loss_percent*available_capital, capital_requirement))
            capital_ratio = capital_requirement_avail/capital_requirement
            capital_requirement *= capital_ratio
            confidence *= capital_ratio
            if capital_requirement < available_capital:
                print('make bet!')
                print('Make bet!')
                print('Odds: ', best_odds1)
                print('Price: ', max_price1)
                print('Prediction: ', prediction)
                print('Tournament', row['tournament'])
                print('Bet on player: ', row['player_id'])
                print('Bet against player: ', row['opponent_id'])

        if max_price2 > 0 and best_odds2 < (1.0 - prediction) - betting_epsilon:
            confidence = (1.0 - prediction - best_odds2) * betting_minimum
            capital_requirement = abs(max_price2) * confidence
            capital_requirement_avail = max(betting_minimum, min(max_loss_percent * available_capital, capital_requirement))
            capital_ratio = capital_requirement_avail / capital_requirement
            capital_requirement *= capital_ratio
            confidence *= capital_ratio
            if capital_requirement < available_capital:
                print('Make bet!')
                print('Odds: ', best_odds2)
                print('Price: ', max_price2)
                print('Prediction', prediction)
                print('Tournament', row['tournament'])
                print('Bet on player: ', row['opponent_id'])
                print('Bet against player: ', row['player_id'])



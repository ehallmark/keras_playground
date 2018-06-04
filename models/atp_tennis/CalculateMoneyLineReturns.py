import keras as k
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import models.atp_tennis.TennisMatchOutcomeNN as tennis_model
from models.atp_tennis.TennisMatchOutcomeNN import test_model,to_percentage

model = k.models.load_model('tennis_match_keras_nn.h5')
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
print(model.summary())

all_data = tennis_model.all_data

test_meta_data = all_data[2]
test_data = all_data[1]
binary_correct, n, binary_percent, avg_error = test_model(model, test_data[0], test_data[1])
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

conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
betting_data = pd.read_sql('''
    select year,tournament,team1,team2,
    min(price1) as min_price1, 
    max(price1) as max_price1,
    min(price2) as min_price2,
    max(price2) as max_price2
    from atp_tennis_betting_link where year=2017 group by year,tournament,team1,team2
''', conn)
betting_epsilon = 0.05
print(betting_data[0:10])
for i in range(test_meta_data.shape[0]):
    row = test_meta_data.iloc[i]
    prediction = predictions[i]
    bet_row = betting_data[
        (betting_data.year == row.year) &
        (betting_data.team1 == row.player_id) &
        (betting_data.team2 == row.opponent_id) &
        (betting_data.tournament == row.tournament)]
    if bet_row.shape[0] > 0:
        # make betting decision
        max_price1 = np.array(bet_row['max_price1']).flatten()[0]
        max_price2 = np.array(bet_row['max_price2']).flatten()[0]
        min_price1 = np.array(bet_row['min_price1']).flatten()[0]
        min_price2 = np.array(bet_row['min_price2']).flatten()[0]
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
        if best_odds1 < prediction - betting_epsilon:
            print('Make BET! Advantage', prediction-best_odds1)
        if best_odds2 < (1.0 - prediction) - betting_epsilon:
            print('Makde BET on OPPONENT! Advantage', (1.0-prediction)-best_odds2)

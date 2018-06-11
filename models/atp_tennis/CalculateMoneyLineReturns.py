import keras as k
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from models.simulation.Simulate import simulate_money_line
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


def parameter_update_func(parameters):
    parameters['max_loss_percent'] = 0.05
    parameters['betting_epsilon1'] = float(0.175 + (np.random.rand(1) * 0.05 - 0.025))
    parameters['betting_epsilon2'] = float(0.20 + (np.random.rand(1) * 0.05 - 0.025))
    parameters['max_price_plus'] = 300.
    parameters['max_price_minus'] = -250.


price_str = 'max_price'
num_trials = 50
parameters = {}


def betting_epsilon_func(price):
    if price > 0:
        return parameters['betting_epsilon1']
    else:
        return parameters['betting_epsilon2']


def predictor_func(i):
    return predictions[0][i]


def actual_label_func(i):
    return test_labels[0][i]


simulate_money_line(predictor_func, actual_label_func, parameter_update_func,
                    betting_epsilon_func, test_meta_data, betting_data, parameters,
                    price_str, num_trials)


exit(0)

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
sql = pd.read_sql('''
    select atp_tennis_betting_link
''')
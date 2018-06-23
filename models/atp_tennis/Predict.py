import numpy as np
import pandas as pd
from random import shuffle
from models.simulation.Simulate import simulate_spread
import models.atp_tennis.TennisMatchSpreadSklearnModels as tennis_model
from models.atp_tennis.TennisMatchOutcomeSklearnModels import load_spread_model, load_outcome_model
from models.atp_tennis.TennisMatchOutcomeNN import test_model,to_percentage

historical_model = load_outcome_model('Logistic')
historical_spread_model = load_spread_model('Linear')

test_year = 2018  # IMPORTANT!!
data, test_data = tennis_model.load_data(start_year=tennis_model.start_year, num_test_years=1, test_year=test_year,
                            test_tournament='london', model=historical_model, spread_model=historical_spread_model)

print('Test data: ', test_data[0:10])
print('Test data shape: ', test_data.shape)

predictions = np.array(tennis_model.predict(data, test_data, train=False)).flatten()
print("predictions shape: ", predictions.shape)

# run betting algo
epsilon = 0.575
test_return, num_bets = simulate_spread(lambda j: predictions[j],
                                        lambda j: test_data['spread'].iloc[j],
                                        tennis_model.bet_func(epsilon), test_data,
                                        'price', 5, sampling=0, shuffle=True, verbose=True)

print('Num bets total: ', len(bets_to_make))
print('Bet On, Bet Against, Confidence, Amount to Invest, Current Price')
for bet_to_make in bets_to_make:
    print(','.join([bet_to_make[0], bet_to_make[1], str(bet_to_make[2]), str(bet_to_make[3]), str(bet_to_make[4])]))



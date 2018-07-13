import numpy as np
import pandas as pd
import datetime
from random import shuffle
import statsmodels.formula.api as smf
from models.simulation.Simulate import simulate_money_line
import models.atp_tennis.TennisMatchBettingSklearnModels as tennis_model
from models.atp_tennis.TennisMatchOutcomeSklearnModels import load_spread_model, load_outcome_model
from models.atp_tennis.TennisMatchOutcomeLogit import test_model,to_percentage

historical_model = load_outcome_model('Logistic0')
historical_spread_model = load_spread_model('Linear0')

historical_model_slam = load_outcome_model('Logistic1')
historical_spread_model_slam = load_spread_model('Linear1')

future_matches_only = False
test_year = 2018  # IMPORTANT!!

data, test_data = tennis_model.load_data(start_year=tennis_model.start_year, num_test_years=1, test_year=test_year,
                                         model=historical_model, spread_model=historical_spread_model,
                                         slam_model=historical_model_slam, slam_spread_model=historical_spread_model_slam)
# print('Test data: ', test_data[0:10])
print('Test data shape: ', test_data.shape)

# regular model
print('Regular model')
y = 'y'
results = smf.logit(y + ' ~ ' + '+'.join(tennis_model.betting_input_attributes), data=data).fit()
print(results.summary())
_, avg_error = test_model(results, test_data, test_data[y], include_binary=False)
print('Average error: ', avg_error)


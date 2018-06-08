import keras as k
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from random import shuffle
import models.atp_tennis.TennisMatchOutcomeNN as tennis_model
from models.atp_tennis.TennisMatchOutcomeNN import test_model,to_percentage,input_attributes


all_data = tennis_model.get_all_data(2017,2017,tournament='roland-garros')
all_data_2018 = tennis_model.get_all_data(2018,2018, tournament='roland-garros')

test_data = all_data[1][0]
test_data_2018 = all_data_2018[1][0]

means = test_data.mean(axis=0)
means_2018 = test_data_2018.mean(axis=0)
var = test_data.var(axis=0)
vars_2018 = test_data_2018.var(axis=0)

average_diff = means_2018-means
var_diff = vars_2018 - var

print('Avg avg diff: ', np.abs(average_diff).mean())
print('Var avg diff: ', np.abs(var_diff).mean(0))
for i in range(len(input_attributes)):
    print('Attr name: ', input_attributes[i])
    print('Mean:', means[i], ' Mean2:', means_2018[i])
    print('Avg diff: ', average_diff[i])
    print('Var diff: ', var_diff[i])

if np.abs(var_diff).mean(0)>1000:
    print('WARNING: MODEL LIKELY HAS ERRORS. VERY LARGE VARIANCE BETWEEN SAMPLES...')
elif np.abs(average_diff).mean() > 10:
    print('WARNING: MODEL LIKELY HAS ERRORS. VERY LARGE MEAN DIFFERENCE BETWEEN SAMPLES...')
else:
    print('PASSED VARIANCE TEST AND MEAN TEST!')

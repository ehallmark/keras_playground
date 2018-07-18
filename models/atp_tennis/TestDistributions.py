import keras as k
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from random import shuffle
import matplotlib.pyplot as plt
import models.atp_tennis.TennisMatchBettingSklearnModels as tennis_model
from models.atp_tennis.TennisMatchBettingSklearnModels import betting_input_attributes,all_attributes,test_model,to_percentage
from scipy import stats
np.random.seed(12345678)  #fix random seed to get the same result


def ks_test(x1, x2):
    return stats.ks_2samp(x1, x2)[1]


def norm_test(x1, x2):
    std = np.std(x1)
    mean = np.mean(x1)
    g2 = (x2 - mean)/std
    return np.mean(np.abs(g2))


models = tennis_model.models
test_data, test_data_2018 = tennis_model.load_data(start_year=2011, test_year=2018, num_test_years=1, models=models, test_tournament='wimbledon', spread_models=None)

print('Test:', test_data[0:10])
print('2018: ', test_data_2018[0:10])

attributes = list(betting_input_attributes)

test_data, test_data_2018 = test_data[attributes], test_data_2018[attributes]
means = test_data.mean(axis=0)
means_2018 = test_data_2018.mean(axis=0)
var = test_data.var(axis=0)
vars_2018 = test_data_2018.var(axis=0)

average_diff = (means_2018-means)
var_diff = (vars_2018 - var)/means

print('Avg avg diff: ', np.abs(average_diff).mean())
print('Var avg diff: ', np.abs(var_diff).mean(0))

for i in range(len(attributes)):
    plt.figure(figsize=(10, 10))
    print('Attr name: ', attributes[i])
    print('Mean:', means[i], ' Mean2:', means_2018[i])
    print('Avg diff: ', average_diff[i])
    print('Var diff/mean: ', var_diff[i])
    print('Norm test: ', norm_test(test_data[attributes[i]], test_data_2018[attributes[i]]))
    minx = np.min(test_data[attributes[i]])
    maxx = np.max(test_data[attributes[i]])
    plt.hist(test_data[attributes[i]], range=(minx, maxx), bins=20, label='Train',
             histtype="step", lw=2)
    plt.hist(test_data_2018[attributes[i]], range=(minx, maxx), bins=20, label='Test',
             histtype="step", lw=2)

    plt.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

if np.abs(var_diff).mean(0)>1000:
    print('WARNING: MODEL LIKELY HAS ERRORS. VERY LARGE VARIANCE BETWEEN SAMPLES...')
elif np.abs(average_diff).mean() > 10:
    print('WARNING: MODEL LIKELY HAS ERRORS. VERY LARGE MEAN DIFFERENCE BETWEEN SAMPLES...')
else:
    print('PASSED VARIANCE TEST AND MEAN TEST!')

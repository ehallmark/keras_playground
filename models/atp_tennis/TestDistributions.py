import keras as k
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from random import shuffle
import models.atp_tennis.TennisMatchMoneyLineSklearnModels as tennis_model
from models.atp_tennis.TennisMatchMoneyLineSklearnModels import betting_input_attributes,all_attributes,outcome_input_attributes,test_model,to_percentage

for outcome_model_name in ['Logistic', 'Naive Bayes']:
    outcome_model = tennis_model.load_outcome_model(outcome_model_name)
    spread_model = tennis_model.load_spread_model('Linear')
<<<<<<< HEAD:models/atp_tennis/TestDistributions.py
    test_data, test_data_2018 = tennis_model.load_data(start_year=2010, test_year=2018, num_test_years=1)
=======
    test_data, test_data_2018 = tennis_model.load_data(outcome_model, spread_model, start_year=2010, test_year=2018, num_test_years=1)
>>>>>>> 441da9c10f3a7cf5ddd0b237d942ea371bf30ad5:models/atp_tennis/TestDistributions.py

    attributes = list(outcome_input_attributes)
    attributes.append('returns')
    for attr in betting_input_attributes:
        if attr not in attributes:
            attributes.append(attr)

    test_data, test_data_2018 = test_data[attributes], test_data_2018[attributes]
    means = test_data.mean(axis=0)
    means_2018 = test_data_2018.mean(axis=0)
    var = test_data.var(axis=0)
    vars_2018 = test_data_2018.var(axis=0)

    average_diff = means_2018-means
    var_diff = vars_2018 - var

    print('Avg avg diff: ', np.abs(average_diff).mean())
    print('Var avg diff: ', np.abs(var_diff).mean(0))
    for i in range(len(attributes)):
        print('Attr name: ', attributes[i])
        print('Mean:', means[i], ' Mean2:', means_2018[i])
        print('Avg diff: ', average_diff[i])
        print('Var diff: ', var_diff[i])

    if np.abs(var_diff).mean(0)>1000:
        print('WARNING: MODEL LIKELY HAS ERRORS. VERY LARGE VARIANCE BETWEEN SAMPLES...')
    elif np.abs(average_diff).mean() > 10:
        print('WARNING: MODEL LIKELY HAS ERRORS. VERY LARGE MEAN DIFFERENCE BETWEEN SAMPLES...')
    else:
        print('PASSED VARIANCE TEST AND MEAN TEST!')

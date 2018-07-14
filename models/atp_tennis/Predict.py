import numpy as np
import pandas as pd
import datetime
from random import shuffle
from models.simulation.Simulate import simulate_money_line
import models.atp_tennis.TennisMatchBettingSklearnModels as tennis_model

historical_model = tennis_model.historical_model
historical_spread_model = tennis_model.historical_spread_model

historical_model_slam = tennis_model.historical_model_slam
historical_spread_model_slam = tennis_model.historical_spread_model_slam

test_year = 2018  # IMPORTANT!!
tournaments = ['roland-garros', 'geneva', 'antalya', 'wimbledon']

predict_for_real = True
for tournament in tournaments:
    print("Tournament: ", tournament)
    data, test_data = tennis_model.load_data(start_year=tennis_model.start_year, num_test_years=1, test_year=test_year,
                                             test_tournament=tournament, model=historical_model, spread_model=historical_spread_model,
                                             slam_model=historical_model_slam, slam_spread_model=historical_spread_model_slam)
    # print('Test data: ', test_data[0:10])
    n2 = test_data.shape[0]
    if not predict_for_real:
        test_data = test_data[np.isfinite(test_data['y'])]

    print('Test data shape: ', test_data.shape)
    # print('Test data: ', test_data['y'])
    n2 = n2 - test_data.shape[0]
    print('Num NaNs: ', n2)

    predictions = np.array(tennis_model.predict(data, test_data, train=False)).flatten()
    print("predictions shape: ", predictions.shape)

    bets_to_make = []

    def after_bet_func(conf, player1, player2, spread, price, amount, date, bookie):
        # print("MAKE BET: ", conf, player1, player2)
        bets_to_make.append([player1, player2, conf, amount, spread, price, date, bookie])

    # run betting algo
    epsilon = 0.0
    test_return, num_bets = simulate_money_line(lambda j: predictions[j],
                                                # lambda j: 0,
                                                # lambda j: 0,
                                                lambda j: test_data['y'].iloc[j],
                                                lambda j: test_data['spread'].iloc[j],
                                                lambda j: 0,
                                                tennis_model.decision_func(epsilon, bet_totals=False),
                                                test_data,
                                                'max_price',
                                                'price',
                                                'totals_price',
                                                1,
                                                after_bet_function=after_bet_func,
                                                sampling=0,
                                                initial_capital=10000000,
                                                shuffle=True, verbose=False)

    print('Num bets total: ', len(bets_to_make))
    print('Num bets: ', num_bets)
    print('Bet On, Bet Against, Confidence, Amount to Invest, Current Spread, Current Price, Date, Book Name')
    bets_to_make.sort(key=lambda x: x[7])  # by book name
    bets_to_make.sort(key=lambda x: x[6])  # by date
    for bet_to_make in bets_to_make:
        print(','.join([bet_to_make[0], bet_to_make[1], str(bet_to_make[2]), str(bet_to_make[3]), str(bet_to_make[4]), str(bet_to_make[5]), str(bet_to_make[6]), str(bet_to_make[7])]))



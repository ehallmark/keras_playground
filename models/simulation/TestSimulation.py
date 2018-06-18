from models.simulation.Simulate import simulate_spread
from models.atp_tennis.TennisMatchMoneyLineSklearnModels import bet_func


if __name__ == '__main__':

    # test edge cases
    test_return, num_bets = simulate_spread(lambda j: prob_pos[j], lambda j: test_data['actual'][j],
                                            lambda j: test_data['spread_actual'][j], lambda _: None,
                                            bet_func(model_to_epsilon[name]), test_data, parameters,
                                            'price', num_tests, sampling=0, shuffle=False)
    print('Final test return:', test_return, ' Num bets:', num_bets, ' Avg Error:', to_percentage(avg_error))
    print('---------------------------------------------------------')

    # test general cases


    # more tests


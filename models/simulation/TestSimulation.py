from models.simulation.Simulate import simulate_money_line
from models.atp_tennis.TennisMatchMoneyLineSklearnModels import bet_func
import pandas as pd

if __name__ == '__main__':

    # test edge cases
    prob_pos = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    actual = [1, 1, 0, 1, 0, 1]
    parameters = {
        'max_loss_percent': 0.05
    }
    test_data = pd.DataFrame(data =
                             {'max_price1': [100, 100, 100, 100, 100, 100],
                              'max_price2': [-100, -100, -100, -100, -100, -100],
                              'field': [1,2,3,4,5,6],
                              'field2': ['something', 'else', 'id', '','as', '']
                              })
    test_return, num_bets = simulate_money_line(lambda j: prob_pos[j], lambda j: actual[j], lambda _: None,
                                                bet_func(0), test_data, parameters,
                                                'max_price', 5, sampling=0, shuffle=True, verbose=False)
    print('Final test return:', test_return, ' Num bets:', num_bets)
    print('---------------------------------------------------------')

    # test general cases
    print('Test data:', test_data)
    test_data.sort_values(by=['field2'], kind='mergesort', inplace=True)
    print('Sorted test data:', test_data)
    test_data.reset_index(drop=True)
    print('Test data reset index:', test_data)
    print('Sorted row 0:', test_data['field'][0])

    # more tests


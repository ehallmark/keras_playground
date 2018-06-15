from models.atp_tennis.TennisMatchOutcomeLogit import test_model, to_percentage
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import models.atp_tennis.TennisMatchOutcomeLogit as tennis_model
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
from sqlalchemy import create_engine
import pandas as pd


input_attributes = [
        'prev_h2h2_wins_player',
        'prev_h2h2_wins_opponent',
        'mean_duration',
        'mean_opp_duration',
        'mean_return_points_made',
        'mean_opp_return_points_made',
        'mean_second_serve_points_made',
        'mean_opp_second_serve_points_made',
        'h2h_prior_win_percent',
        'prev_year_prior_encounters',
        'opp_prev_year_prior_encounters',
        'prev_year_avg_round',
        'opp_prev_year_avg_round',
        'opp_tourney_hist_avg_round',
        'tourney_hist_avg_round',
        'tourney_hist_prior_encounters',
        'opp_tourney_hist_prior_encounters',
        #'mean_break_points_made',
        #'mean_opp_break_points_made',
        #'previous_tournament_round',
        #'opp_previous_tournament_round',
        'tiebreak_win_percent',
        'opp_tiebreak_win_percent',
        'surface_experience',
        'opp_surface_experience',
        'experience',
        'opp_experience',
        'age',
        'opp_age',
        #'lefty',
        #'opp_lefty',
        #'weight',
        #'opp_weight',
        'height',
        'opp_height',
        'duration_prev_match',
        'opp_duration_prev_match',
        'elo_score',
        'opp_elo_score',
        'odds1',
        'predictions'
    ]

y_str = 'place_bet'
all_attributes = list(input_attributes)
all_attributes.append(y_str)
meta_attributes = ['player_id', 'opponent_id', 'tournament', 'year']
for meta in meta_attributes:
    all_attributes.append(meta)


def load_outcome_predictions_and_actuals(model, test_year=2018, start_year=2005):
    all_data = tennis_model.get_all_data(test_season=test_year, start_year=start_year)
    data, meta_data = all_data[0], all_data[1]
    test_data, test_meta_data = all_data[2], all_data[3]
    labels = data[1]
    data = data[0]
    test_labels = test_data[1]
    test_data = test_data[0]
    X = np.array(data[input_attributes])
    y = np.array(data[y_str]).flatten()
    model.fit(X, y)
    avg_error = test_model(model, test_data, test_labels)
    print('Average error: ', to_percentage(avg_error))
    print('Test Meta Data Size: ', test_meta_data.shape[0])
    data['predictions'] = model.predict(data)
    data[y] = labels
    meta_data['labels'] = labels
    test_data['predictions'] = model.predict(test_data)
    test_meta_data['labels'] = test_labels
    test_data[y] = test_labels
    return data, meta_data, test_data, test_meta_data


def load_betting_data(betting_sites, test_year=2018):
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
     '''.replace('{{YEAR}}', str(test_year)).replace('{{BOOK_NAMES}}', '\'' + '\',\''.join(betting_sites) + '\''), conn)
    return betting_data


def load_data(model, test_year):
    data, meta_data, test_data, test_meta_data = load_outcome_predictions_and_actuals(model, test_year=test_year)
    betting_sites = ['Bovada', '5Dimes', 'BetOnline']
    betting_data = load_betting_data(betting_sites, test_year=test_year)
    original_len = test_meta_data.shape[0]
    test_meta_data = pd.DataFrame.merge(
        test_meta_data,
        betting_data,
        'inner',
        left_on=['year', 'player_id', 'opponent_id', 'tournament'],
        right_on=['year', 'team1', 'team2', 'tournament'])
    after_len = test_meta_data.shape[0]
    if original_len != after_len:
        print('Join data has different length... ', original_len, after_len)
        exit(1)
    meta_data = pd.DataFrame.merge(
        meta_data,
        betting_data,
        'inner',
        left_on=['year', 'player_id', 'opponent_id', 'tournament'],
        right_on=['year', 'team1', 'team2', 'tournament'])
    return data, meta_data, test_data, test_meta_data


if __name__ == '__main__':
    test_year = 2017
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier(n_estimators=500)
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for model, name in [(lr, 'Logistic'),
                      (gnb, 'Naive Bayes'),
                      (svc, 'Support Vector Classification'),
                      (rfc, 'Random Forest')]:
        data, meta_data, test_data, meta_data_test = load_data(model, test_year=test_year)
        print('Input Attrs: ', data[input_attributes][0:5])
        X_test = np.array(test_data[input_attributes])
        y_test = np.array(test_data[y_str]).flatten()

        binary_correct, n, binary_percent, avg_error = test_model(model, X_test, y_test)

        print('Correctly predicted: '+str(binary_correct)+' out of '+str(n) +
              ' ('+to_percentage(binary_percent)+')')
        print('Average error: ', to_percentage(avg_error))
        #model..save(model_file)
        if hasattr(model, "predict_proba"):
            prob_pos = model.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = model.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name,))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()


from models.atp_tennis.TennisMatchOutcomeLogit import load_data, all_attributes, input_attributes_spread, input_attributes0, test_model, to_percentage
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
from sklearn.externals import joblib

outcome_model_file = 'tennis_match_outcome_model_'
spread_model_file = 'tennis_match_spread_model_'


def load_outcome_model(model_name):
    return joblib.load(outcome_model_file+model_name.lower().replace(' ','_'))


def load_spread_model(model_name):
    return joblib.load(spread_model_file+model_name.lower().replace(' ','_'))


def save_outcome_model(model, model_name):
    joblib.dump(model, outcome_model_file+model_name.lower().replace(' ','_'))


def save_spread_model(model, model_name):
    joblib.dump(model, spread_model_file+model_name.lower().replace(' ','_'))


if __name__ == '__main__':
    data, data_test = load_data(all_attributes, test_season=2012, start_year=1995)
    #slam_data = data[data.grand_slam > 0.5]
    #data = data[data.grand_slam < 0.5]
    #slam_data_test = data_test[data_test.grand_slam > 0.5]
    #data_test = data_test[data_test.grand_slam < 0.5]

    datasets = [
        (data, data_test, 'All'),
        (data[data.tournament_rank < 1000], data_test[data_test.tournament_rank < 1000], '500'),
        (data[data.tournament_rank == 1000], data_test[data_test.tournament_rank == 1000], '1000'),
        (data[data.tournament_rank == 2000], data_test[data_test.tournament_rank == 2000], '2000'),
        (data[data.clay > 0.5], data_test[data_test.clay > 0.5], 'Clay'),
        (data[data.grass > 0.5], data_test[data_test.grass > 0.5], 'Grass'),
        (data[data.hard > 0.5], data_test[data_test.hard > 0.5], 'Hard'),
    ]

    train_outcome_model = True
    train_spread_model = True
    if train_outcome_model:
        lr = LogisticRegression()
        gnb = GaussianNB()
        #svc = LinearSVC(C=1.0)
        # rfc = RandomForestClassifier(n_estimators=300)
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for model, name in [(lr, 'Logistic'),
                            (gnb, 'Naive Bayes'),
                            #(svc, 'Support Vector Classification'),
                            # (rfc, 'Random Forest')
                            ]:
            y_str = 'y'
            i = 0
            #for sql, test_data in [(data, data_test), (slam_data, slam_data_test)]:
            for sql, test_data, dataset_name in datasets:
                y = np.array(sql[y_str]).flatten()
                y_test = np.array(test_data[y_str]).flatten()
                attrs = input_attributes0
                X = np.array(sql[attrs])
                X_test = np.array(test_data[attrs])
                model.fit(X, y)
                print('Fit dataset '+str(i)+':', dataset_name, ' Shape:', sql.shape)
                save_outcome_model(model, name+dataset_name)
                model = load_outcome_model(name+dataset_name)
                print('Saved and reloaded.')
                binary_correct, n, binary_percent, avg_error = test_model(model, X_test, y_test)

                if name == 'Logistic':
                    results = smf.logit(y_str + ' ~ ' + '+'.join(attrs), data=sql).fit()
                    print(results.summary())

                print('Correctly predicted: '+str(binary_correct)+' out of '+str(n) +
                      ' ('+to_percentage(binary_percent)+')')
                print('Average error: ', to_percentage(avg_error))
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
                i += 1

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        plt.show()

    if train_spread_model:
        lr = LinearRegression()
        #rf = RandomForestRegressor(n_estimators=30)
        plt.figure(figsize=(10, 10))
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        for model, name in [(lr, 'Linear'),
                            #(rf, 'Random Forest')
                            ]:
            y_str = 'spread'
            i = 0
            for sql, test_data in [(data, data_test), (slam_data, slam_data_test)]:
                X = np.array(sql[input_attributes_spread])
                y = np.array(sql[y_str]).flatten()
                X_test = np.array(test_data[input_attributes_spread])
                y_test = np.array(test_data[y_str]).flatten()
                model.fit(X, y)
                print('Fit.')
                save_spread_model(model, name+str(i))
                model = load_spread_model(name+str(i))
                print('Saved and reloaded.')
                n, avg_error = test_model(model, X_test, y_test, include_binary=False)
                print('Average error: ', avg_error)
                spread_predictions = model.predict(X_test)
                print('Predictions spread: ', spread_predictions)

                ax2.hist(spread_predictions, range=(min(spread_predictions), max(spread_predictions)), bins=10, label=name,
                         histtype="step", lw=2)
                i += 1

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.show()


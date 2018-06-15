from models.atp_tennis.TennisMatchOutcomeLogit import load_data, all_attributes, input_attributes, test_model, to_percentage
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
from sklearn.externals import joblib

if __name__ == '__main__':
    model_file = 'tennis_match_outcome_model_'
    sql, test_data = load_data(all_attributes, test_season=2010, start_year=1996)
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    #rfc = RandomForestClassifier(n_estimators=300)
    print('Attrs: ', sql[all_attributes][0:20])
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for model, name in [(lr, 'Logistic'),
                        (gnb, 'Naive Bayes'),
                        (svc, 'Support Vector Classification'),
                        #(rfc, 'Random Forest')
                        ]:
        y_str = 'y'
        X = np.array(sql[input_attributes])
        y = np.array(sql[y_str]).flatten()
        X_test = np.array(test_data[input_attributes])
        y_test = np.array(test_data[y_str]).flatten()
        model.fit(X, y)
        file_name = model_file+name.lower().replace(' ', '_')
        print('Fit.')
        joblib.dump(model, file_name)
        model = joblib.load(file_name)
        print('Saved and reloaded.')
        binary_correct, n, binary_percent, avg_error = test_model(model, X_test, y_test)

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

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()


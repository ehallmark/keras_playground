import pandas as pd
from sqlalchemy import create_engine
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
'''

'''


def test_model(model, x, y):
    predictions = model.predict(x)
    binary_predictions = (predictions >= 0.5).astype(int)
    binary_errors = np.array(y) != np.array(binary_predictions)
    binary_errors = binary_errors.astype(int)
    errors = np.array(y - np.array(predictions))
    binary_correct = predictions.shape[0] - int(binary_errors.sum())
    binary_percent = float(binary_correct) / predictions.shape[0]
    avg_error = np.mean(np.abs(errors), -1)
    return binary_correct, y.shape[0], binary_percent, avg_error


def to_percentage(x):
    return str("%0.2f" % (x * 100))+'%'


def bool_to_int(b):
    if b:
        return 1.0
    else:
        return 0.0


conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
sql = pd.read_sql('''
    select 
        case when m.player_victory then 1.0 else 0.0 end as y, 
        case when court_surface = 'Clay' then 1.0 else 0.0 end as clay,
        case when court_surface = 'Grass' then 1.0 else 0.0 end as grass,
        m.year,
        coalesce(h2h.prior_encounters,0) as h2h_prior_encounters,
        coalesce(h2h.prior_victories,0) as h2h_prior_victories,
        coalesce(h2h.prior_losses,0) as h2h_prior_losses,
        coalesce(h2h.prior_victories,0)-coalesce(h2h.prior_losses,0) as h2h_prior_win_percent,        
        coalesce(prev_year.prior_encounters,0) as prev_year_prior_encounters,
        coalesce(prev_year.prior_victories,0) as prev_year_prior_victories,
        coalesce(prev_year.prior_losses,0) as prev_year_prior_losses,
        coalesce(prev_year.prior_victories,0)-coalesce(prev_year.prior_losses,0) as prev_year_prior_win_percent,        
        coalesce(tourney_hist.prior_encounters,0) as tourney_hist_prior_encounters,
        coalesce(tourney_hist.prior_victories,0) as tourney_hist_prior_victories,
        coalesce(tourney_hist.prior_victories,0)-coalesce(tourney_hist.prior_losses,0) as tourney_hist_prior_win_percent,
        coalesce(tourney_hist.prior_losses,0) as tourney_hist_prior_losses,
        coalesce(prev_year_opp.prior_encounters,0) as opp_prev_year_prior_encounters,
        coalesce(prev_year_opp.prior_victories,0) as opp_prev_year_prior_victories,
        coalesce(prev_year_opp.prior_losses,0) as opp_prev_year_prior_losses,
        coalesce(prev_year_opp.prior_victories,0)-coalesce(prev_year_opp.prior_losses,0) as opp_prev_year_prior_win_percent,        
        coalesce(tourney_hist_opp.prior_encounters,0) as opp_tourney_hist_prior_encounters,
        coalesce(tourney_hist_opp.prior_victories,0) as opp_tourney_hist_prior_victories,
        coalesce(tourney_hist_opp.prior_victories,0)-coalesce(tourney_hist_opp.prior_losses,0) as opp_tourney_hist_prior_win_percent,
        coalesce(tourney_hist_opp.prior_losses,0) as opp_tourney_hist_prior_losses,
        coalesce(mean.first_serve_points_made, 0) as mean_first_serve_points_made,
        coalesce(mean_opp.first_serve_points_made, 0) as mean_opp_first_serve_points_made,
        coalesce(mean.second_serve_points_made, 0) as mean_second_serve_points_made,
        coalesce(mean_opp.second_serve_points_made, 0) as mean_opp_second_serve_points_made,
        coalesce(mean.return_points_won, 0) as mean_return_points_made,
        coalesce(mean_opp.return_points_won, 0) as mean_opp_return_points_made 
    from atp_matches_individual as m
    left outer join atp_matches_prior_h2h as h2h 
        on ((m.player_id,m.opponent_id,m.tournament,m.year)=(h2h.player_id,h2h.opponent_id,h2h.tournament,h2h.year))
    left outer join atp_matches_prior_year as prev_year
        on ((m.player_id,m.tournament,m.year)=(prev_year.player_id,prev_year.tournament,prev_year.year))
    left outer join atp_matches_tournament_history as tourney_hist
        on ((m.player_id,m.tournament,m.year)=(tourney_hist.player_id,tourney_hist.tournament,tourney_hist.year))
    left outer join atp_matches_prior_year as prev_year_opp
        on ((m.opponent_id,m.tournament,m.year)=(prev_year_opp.player_id,prev_year_opp.tournament,prev_year_opp.year))
    left outer join atp_matches_tournament_history as tourney_hist_opp
        on ((m.opponent_id,m.tournament,m.year)=(tourney_hist_opp.player_id,tourney_hist_opp.tournament,tourney_hist_opp.year))
    left outer join atp_matches_prior_year_avg as mean
        on ((m.player_id,m.tournament,m.year)=(mean.player_id,mean.tournament,mean.year))
    left outer join atp_matches_prior_year_avg as mean_opp
        on ((m.opponent_id,m.tournament,m.year)=(mean_opp.player_id,mean_opp.tournament,mean_opp.year))    
    where m.year <= 2017 and m.year >= 2005
    and m.first_serve_attempted > 0
''', conn)

test_season = 2017

print('Data: ', sql[:10])

input_attributes = [
    #'clay',
    #'grass',
    #'mean_return_points_made',
    #'mean_opp_return_points_made',
    'mean_second_serve_points_made',
    'mean_opp_second_serve_points_made',
    #'mean_first_serve_points_made',
    #'mean_opp_first_serve_points_made',
    'h2h_prior_win_percent',
    #'h2h_prior_encounters',
    #'h2h_prior_victories',
    #'prev_year_prior_win_percent',
    'prev_year_prior_encounters',
    'opp_prev_year_prior_encounters',
    #'tourney_hist_prior_win_percent',
    #'tourney_hist_prior_victories',
    'tourney_hist_prior_encounters',
    'opp_tourney_hist_prior_encounters',
    #'first_serve_made',
    #'first_serve_attempted',
    #'return_points_won',
    #'return_points_attempted'
]
all_attributes = list(input_attributes)
all_attributes.append('y')
all_attributes.append('year')

sql = sql[all_attributes].astype(np.float64)

test_data = sql[sql.year == test_season]
sql = sql[sql.year != test_season]

print('Attrs: ', sql[input_attributes])

# model to predict the total score (h_pts + a_pts)
results = smf.logit('y ~ '+'+'.join(input_attributes), data=sql).fit()
print(results.summary())

binary_correct, n, binary_percent, avg_error = test_model(results, test_data, test_data['y'])

print('Correctly predicted: '+str(binary_correct)+' out of '+str(n) +
      ' ('+to_percentage(binary_percent)+')')
print('Average error: ', to_percentage(avg_error))

exit(0)


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


def load_data(attributes, test_season=2017):
    conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
    sql = pd.read_sql('''
        select 
            case when m.player_victory then 1.0 else 0.0 end as y, 
            case when court_surface = 'Clay' then 1.0 else 0.0 end as clay,
            case when court_surface = 'Grass' then 1.0 else 0.0 end as grass,
            m.year as year,
            m.player_id as player_id,
            m.opponent_id as opponent_id,
            m.tournament as tournament,
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
            coalesce(mean_opp.return_points_won, 0) as mean_opp_return_points_made,
            coalesce(mean.break_points_against, 0) as mean_break_points_against,
            coalesce(mean_opp.break_points_against, 0) as mean_opp_break_points_against,
            coalesce(mean.break_points_saved, 0) as mean_break_points_saved,
            coalesce(mean_opp.break_points_saved, 0) as mean_opp_break_points_saved,
            coalesce(mean.break_points_attempted, 0) as mean_break_points_attempted,
            coalesce(mean_opp.break_points_attempted, 0) as mean_opp_break_points_attempted,
            coalesce(mean.break_points_made, 0) as mean_break_points_made,
            coalesce(mean_opp.break_points_made, 0) as mean_opp_break_points_made,
            coalesce(prior_tourney.round, 0) as previous_tournament_round,
            coalesce(prior_tourney_opp.round, 0) as opp_previous_tournament_round,
            coalesce(tb.tiebreaks_won,0)::float/(1+coalesce(tb.tiebreaks_total,0)) as tiebreak_win_percent,
            coalesce(tb_opp.tiebreaks_won,0)::float/(1+coalesce(tb_opp.tiebreaks_total,0)) as opp_tiebreak_win_percent,
            case when m.court_surface='Clay' then coalesce(se.clay,0)::float/(1+coalesce(se.total_matches))
            else case when m.court_surface='Grass' then coalesce(se.grass,0)::float/(1+coalesce(se.total_matches))
            else coalesce(se.hard,0)::float/(1+coalesce(se.total_matches)) end end as surface_experience,
            case when m.court_surface='Clay' then coalesce(se_opp.clay,0)::float/(1+coalesce(se_opp.total_matches))
            else case when m.court_surface='Grass' then coalesce(se_opp.grass,0)::float/(1+coalesce(se_opp.total_matches))
            else coalesce(se_opp.hard,0)::float/(1+coalesce(se_opp.total_matches)) end end as opp_surface_experience,
            case when coalesce(pc.left_handed,false) then 1.0 else 0.0 end as lefty,
            case when coalesce(pc_opp.left_handed,false) then 1.0 else 0.0 end as opp_lefty,
            coalesce(pc.height,(select avg_height from avg_player_characteristics)) as height,
            coalesce(pc_opp.height,(select avg_height from avg_player_characteristics)) as opp_height,
            coalesce(pc.weight,(select avg_weight from avg_player_characteristics)) as weight,
            coalesce(pc_opp.weight,(select avg_weight from avg_player_characteristics)) as opp_weight,
            coalesce(var.first_serve_percent,0.5) as var_first_serve_percent,
            coalesce(var_opp.first_serve_percent,0.5) as opp_var_first_serve_percent,
            coalesce(var.first_serve_points_percent,0.5) as var_first_serve_points_percent,
            coalesce(var_opp.first_serve_points_percent,0.5) as opp_var_first_serve_points_percent,
            coalesce(var.second_serve_points_percent,0.5) as var_second_serve_points_percent,
            coalesce(var_opp.second_serve_points_percent,0.5) as opp_var_second_serve_points_percent,
            coalesce(var.break_points_saved_percent,0.5) as var_break_points_saved_percent,
            coalesce(var_opp.break_points_saved_percent,0.5) as opp_var_break_points_saved_percent,
            coalesce(var.first_serve_return_points_percent,0.5) as var_first_serve_return_points_percent,
            coalesce(var_opp.first_serve_return_points_percent,0.5) as opp_var_first_serve_return_points_percent,
            coalesce(var.second_serve_return_points_percent,0.5) as var_second_serve_return_points_percent,
            coalesce(var_opp.second_serve_return_points_percent,0.5) as opp_var_second_serve_return_points_percent,
            coalesce(var.break_points_percent,0.5) as var_break_points_percent,
            coalesce(var_opp.break_points_percent,0.5) as opp_var_break_points_percent,
            case when pc.date_of_birth is null then (select avg_age from avg_player_characteristics)
                else m.year - extract(year from pc.date_of_birth) end as age,
            case when pc_opp.date_of_birth is null then (select avg_age from avg_player_characteristics)
                else m.year - extract(year from pc_opp.date_of_birth) end as opp_age,
            case when pc.turned_pro is null then (select avg_experience from avg_player_characteristics)
                else m.year - pc.turned_pro end as experience,
            case when pc_opp.turned_pro is null then (select avg_experience from avg_player_characteristics)
                else m.year - pc_opp.turned_pro end as opp_experience
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
        left outer join atp_matches_prior_year_var as var
            on ((m.player_id,m.tournament,m.year)=(var.player_id,var.tournament,var.year))
        left outer join atp_matches_prior_year_var as var_opp
            on ((m.opponent_id,m.tournament,m.year)=(var_opp.player_id,var_opp.tournament,var_opp.year))   
        left outer join atp_matches_prior_year_tournament_round as prior_tourney
            on ((m.player_id,m.tournament,m.year)=(prior_tourney.player_id,prior_tourney.tournament,prior_tourney.year))
        left outer join atp_matches_prior_year_tournament_round as prior_tourney_opp
            on ((m.opponent_id,m.tournament,m.year)=(prior_tourney_opp.player_id,prior_tourney_opp.tournament,prior_tourney_opp.year)) 
        left outer join atp_matches_prior_tiebreak_percentage as tb
            on ((m.player_id,m.tournament,m.year)=(tb.player_id,tb.tournament,tb.year)) 
        left outer join atp_matches_prior_tiebreak_percentage as tb_opp
            on ((m.opponent_id,m.tournament,m.year)=(tb_opp.player_id,tb_opp.tournament,tb_opp.year)) 
        left outer join atp_matches_prior_surface_experience as se 
            on ((m.player_id,m.tournament,m.year)=(se.player_id,se.tournament,se.year))
        left outer join atp_matches_prior_surface_experience as se_opp
            on ((m.opponent_id,m.tournament,m.year)=(se_opp.player_id,se_opp.tournament,se_opp.year))
        left outer join atp_player_characteristics as pc
            on ((m.player_id,m.tournament,m.year)=(pc.player_id,pc.tournament,pc.year))
        left outer join atp_player_characteristics as pc_opp
            on ((m.opponent_id,m.tournament,m.year)=(pc_opp.player_id,pc_opp.tournament,pc_opp.year))
        where m.year <= {{END_DATE}} and m.year >= 2003 
        and m.first_serve_attempted > 0

    '''.replace('{{END_DATE}}', str(test_season)), conn)
    #print('Data: ', sql[:10])

    sql = sql[attributes].astype(np.float64, errors='ignore')

    test_data = sql[sql.year == test_season]
    sql = sql[sql.year != test_season]
    return sql, test_data


if __name__ == '__main__':

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
        #'return_points_attempted',
        #'mean_break_points_made',
        #'mean_opp_break_points_made',
        'previous_tournament_round',
        'opp_previous_tournament_round',
        'tiebreak_win_percent',
        'opp_tiebreak_win_percent',
        'surface_experience',
        'opp_surface_experience',
        'experience',
        'opp_experience',
        'age',
        'opp_age',
        'lefty',
        'opp_lefty',
        'weight','opp_weight',
        'height','opp_height'
    ]
    all_attributes = list(input_attributes)
    all_attributes.append('y')
    all_attributes.append('year')

    sql, test_data = load_data(all_attributes)
    print('Attrs: ', sql[input_attributes])

    # model to predict the total score (h_pts + a_pts)
    results = smf.logit('y ~ '+'+'.join(input_attributes), data=sql).fit()
    print(results.summary())

    binary_correct, n, binary_percent, avg_error = test_model(results, test_data, test_data['y'])

    print('Correctly predicted: '+str(binary_correct)+' out of '+str(n) +
          ' ('+to_percentage(binary_percent)+')')
    print('Average error: ', to_percentage(avg_error))

    exit(0)


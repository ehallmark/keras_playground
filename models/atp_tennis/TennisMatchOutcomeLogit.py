import pandas as pd
from sqlalchemy import create_engine
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
'''

'''


def test_model(model, x, y, include_binary=True):
    predictions = model.predict(x)
    return score_predictions(predictions, y, include_binary=include_binary)


def score_predictions(predictions, y, include_binary=True):
    errors = np.array(y - np.array(predictions))
    avg_error = np.mean(np.abs(errors))
    if include_binary:
        binary_predictions = (predictions >= 0.5).astype(int)
        binary_errors = np.array(y) != np.array(binary_predictions)
        binary_errors = binary_errors.astype(int)
        binary_correct = predictions.shape[0] - int(binary_errors.sum())
        binary_percent = float(binary_correct) / predictions.shape[0]
        return binary_correct, y.shape[0], binary_percent, avg_error
    else:
        return y.shape[0], avg_error


def to_percentage(x):
    return str("%0.2f" % (x * 100))+'%'


def bool_to_int(b):
    if b:
        return 1.0
    else:
        return 0.0


'''
case when ml.price1 > 0 then 100.0/(100.0 + ml.price1) else -1.0*(ml.price1/(-1.0*ml.price1 + 100.0)) end as ml_odds1,
            case when ml.price2 > 0 then 100.0/(100.0 + ml.price2) else -1.0*(ml.price2/(-1.0*ml.price2 + 100.0)) end as ml_odds2,
            case when m.player_victory and ml.price1 > 0 then ml.price1/100.0
                when m.player_victory and ml.price1 < 0 then 100.0/(-ml.price1)
                when not m.player_victory and ml.price1 > 0 then -1.0
                else -1.0 end as ml_return1,
            case when not m.player_victory and ml.price2 > 0 then ml.price2/100.0
                when not m.player_victory and ml.price2 < 0 then 100.0/(-ml.price2)
                when m.player_victory and ml.price2 > 0 then -1.0
                else -1.0 end as ml_return2,    
'''


def load_data(attributes, test_season=2017, start_year=1996, keep_nulls=False):
    conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
    sql_str = '''
        select 
            (m.games_won+m.games_against)::double precision as totals,
            (m.games_won-m.games_against)::double precision as spread,
            case when m.player_victory is null then null else case when m.player_victory then 1.0 else 0.0 end end as y, 
            case when m.court_surface = 'Clay' then 1.0 else 0.0 end as clay,
            case when m.court_surface = 'Grass' then 1.0 else 0.0 end as grass,
            m.court_surface as court_surface,
            m.year as year,
            r.round as round_num,
            m.player_id as player_id,
            m.opponent_id as opponent_id,
            m.tournament as tournament,
            m.num_sets as num_sets,
            case when m.num_sets > 2 then 1 else 0 end as num_sets_greater_than_2,
            case when greatest(m.num_sets-m.sets_won,m.sets_won)=3 or m.tournament in ('roland-garros','wimbledon','us-open','australian-open')
                then 1.0 else 0.0 end as grand_slam,
            case when coalesce(qualifying.had_qualifier,'f') then 1.0 else 0.0 end as had_qualifier,
            case when coalesce(qualifying_opp.had_qualifier,'f') then 1.0 else 0.0 end as opp_had_qualifier,
            coalesce(majors.prior_encounters,0) as major_encounters,
            coalesce(opp_majors.prior_encounters, 0) as opp_major_encounters,
            coalesce(majors.prior_victories,0) as major_victories,
            coalesce(opp_majors.prior_victories, 0) as opp_major_victories,
            coalesce(majors.avg_round,0) as major_avg_round,
            coalesce(opp_majors.avg_round, 0) as opp_major_avg_round,
            coalesce(majors.avg_games_per_set,0) as major_games_per_set,
            coalesce(opp_majors.avg_games_per_set, 0) as opp_major_games_per_set,
            coalesce(majors.avg_match_closeness,0) as major_match_closeness,
            coalesce(opp_majors.avg_match_closeness, 0) as opp_major_match_closeness,
            coalesce(h2h.prior_encounters,0) as h2h_prior_encounters,
            coalesce(h2h.prior_victories,0) as h2h_prior_victories,
            coalesce(h2h.prior_losses,0) as h2h_prior_losses,
            coalesce(h2h.prior_victories,0)-coalesce(h2h.prior_losses,0) as h2h_prior_win_percent,        
            coalesce(prev_quarter.prior_victories,0)/greatest(1,coalesce(prev_quarter.prior_encounters,0)) as prior_quarter_win_percent,
            coalesce(prev_quarter_opp.prior_victories,0)/greatest(1,coalesce(prev_quarter_opp.prior_encounters,0)) as opp_prior_quarter_win_percent,
            coalesce(prev_quarter.prior_encounters,0) as prior_quarter_encounters,
            coalesce(prev_quarter_opp.prior_encounters,0) as opp_prior_quarter_encounters,
            coalesce(prev_quarter.avg_round,0) as prior_quarter_avg_round,
            coalesce(prev_quarter_opp.avg_round,0) as opp_prior_quarter_avg_round,
            coalesce(prev_year.avg_match_closeness,0) as prior_year_match_closeness,
            coalesce(prev_year_opp.avg_match_closeness,0) as opp_prior_year_match_closeness,
            coalesce(prev_quarter.avg_match_closeness,0) as prior_quarter_match_closeness,
            coalesce(prev_quarter_opp.avg_match_closeness,0) as opp_prior_quarter_match_closeness,
            coalesce(prev_quarter.avg_games_per_set,0) as prior_quarter_games_per_set,
            coalesce(prev_quarter_opp.avg_games_per_set,0) as opp_prior_quarter_games_per_set,
            coalesce(prev_quarter.prior_victories,0) as prior_quarter_victories,
            coalesce(prev_quarter_opp.prior_victories,0) as opp_prior_quarter_victories,
            coalesce(prev_quarter.prior_losses,0) as prior_quarter_losses,
            coalesce(prev_quarter_opp.prior_losses,0) as opp_prior_quarter_losses,
            coalesce(prev_year.prior_encounters,0) as prev_year_prior_encounters,
            coalesce(prev_year.prior_victories,0) as prev_year_prior_victories,
            coalesce(prev_year.prior_losses,0) as prev_year_prior_losses,
            coalesce(prev_year.prior_victories,0)-coalesce(prev_year.prior_losses,0) as prev_year_prior_win_percent,        
            coalesce(prev_year.avg_games_per_set,3.0) as avg_games_per_set,
            coalesce(prev_year_opp.avg_games_per_set,3.0) as opp_avg_games_per_set,
            coalesce(prev_year.avg_spread_per_set,0.0) as avg_spread_per_set,
            coalesce(prev_year_opp.avg_spread_per_set,0.0) as opp_avg_spread_per_set,
            coalesce(tourney_hist.prior_encounters,0) as tourney_hist_prior_encounters,
            coalesce(tourney_hist.avg_round,0) as tourney_hist_avg_round,
            coalesce(prev_year.avg_round,0) as prev_year_avg_round,
            coalesce(tourney_hist.prior_victories,0) as tourney_hist_prior_victories,
            coalesce(tourney_hist.prior_victories,0)-coalesce(tourney_hist.prior_losses,0) as tourney_hist_prior_win_percent,
            coalesce(tourney_hist.prior_losses,0) as tourney_hist_prior_losses,
            coalesce(prev_year_opp.prior_encounters,0) as opp_prev_year_prior_encounters,
            coalesce(prev_year_opp.prior_victories,0) as opp_prev_year_prior_victories,
            coalesce(prev_year_opp.prior_losses,0) as opp_prev_year_prior_losses,
            coalesce(tourney_hist_opp.avg_round,0) as opp_tourney_hist_avg_round,
            coalesce(prev_year_opp.avg_round,0) as opp_prev_year_avg_round,
            coalesce(prev_year_opp.prior_victories,0)-coalesce(prev_year_opp.prior_losses,0) as opp_prev_year_prior_win_percent,        
            coalesce(tourney_hist_opp.prior_encounters,0) as opp_tourney_hist_prior_encounters,
            coalesce(tourney_hist_opp.prior_victories,0) as opp_tourney_hist_prior_victories,
            coalesce(tourney_hist_opp.prior_victories,0)-coalesce(tourney_hist_opp.prior_losses,0) as opp_tourney_hist_prior_win_percent,
            coalesce(tourney_hist_opp.prior_losses,0) as opp_tourney_hist_prior_losses,
            coalesce(mean.duration,90.0) as mean_duration,
            coalesce(mean_opp.duration,90.0) as opp_mean_duration,
            coalesce(var.duration,1000.0) as var_duration,
            coalesce(var_opp.duration,1000.0) as var_opp_duration,
            coalesce(mean.first_serve_points_made, 0) as mean_first_serve_points_made,
            coalesce(mean_opp.first_serve_points_made, 0) as opp_mean_first_serve_points_made,
            coalesce(mean.second_serve_points_made, 0) as mean_second_serve_points_made,
            coalesce(mean_opp.second_serve_points_made, 0) as opp_mean_second_serve_points_made,
            coalesce(mean.return_points_won, 0) as mean_return_points_made,
            coalesce(mean_opp.return_points_won, 0) as opp_mean_return_points_made,
            coalesce(mean.break_points_against, 0) as mean_break_points_against,
            coalesce(mean_opp.break_points_against, 0) as opp_mean_break_points_against,
            coalesce(mean.break_points_saved, 0) as mean_break_points_saved,
            coalesce(mean_opp.break_points_saved, 0) as opp_mean_break_points_saved,
            coalesce(mean.break_points_attempted, 0) as mean_break_points_attempted,
            coalesce(mean_opp.break_points_attempted, 0) as opp_mean_break_points_attempted,
            coalesce(mean.break_points_made, 0) as mean_break_points_made,
            coalesce(mean_opp.break_points_made, 0) as opp_mean_break_points_made,
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
            coalesce(var.first_serve_percent,0.05) as var_first_serve_percent,
            coalesce(var_opp.first_serve_percent,0.05) as opp_var_first_serve_percent,
            coalesce(var.first_serve_points_percent,0.05) as var_first_serve_points_percent,
            coalesce(var_opp.first_serve_points_percent,0.05) as opp_var_first_serve_points_percent,
            coalesce(var.second_serve_points_percent,0.05) as var_second_serve_points_percent,
            coalesce(var_opp.second_serve_points_percent,0.05) as opp_var_second_serve_points_percent,
            coalesce(var.break_points_saved_percent,0.05) as var_break_points_saved_percent,
            coalesce(var_opp.break_points_saved_percent,0.05) as opp_var_break_points_saved_percent,
            coalesce(var.first_serve_return_points_percent,0.05) as var_first_serve_return_points_percent,
            coalesce(var_opp.first_serve_return_points_percent,0.05) as opp_var_first_serve_return_points_percent,
            coalesce(var.second_serve_return_points_percent,0.05) as var_second_serve_return_points_percent,
            coalesce(var_opp.second_serve_return_points_percent,0.05) as opp_var_second_serve_return_points_percent,
            coalesce(var.break_points_percent,0.05) as var_break_points_percent,
            coalesce(var_opp.break_points_percent,0.05) as opp_var_break_points_percent,
            extract(epoch from coalesce(prior_match.duration,'01:30:00'::time))::float/3600.0 as duration_prev_match,
            extract(epoch from coalesce(prior_match_opp.duration,'01:30:00'::time))::float/3600.0 as opp_duration_prev_match,         
            coalesce(prior_match.games_won, 6) + coalesce(prior_match.games_against, 6) as previous_games_total,
            coalesce(prior_match_opp.games_won, 6) + coalesce(prior_match_opp.games_against, 6) as opp_previous_games_total,
            case when pc.date_of_birth is null then (select avg_age from avg_player_characteristics)
                else m.year - extract(year from pc.date_of_birth) end as age,
            case when pc_opp.date_of_birth is null then (select avg_age from avg_player_characteristics)
                else m.year - extract(year from pc_opp.date_of_birth) end as opp_age,
            case when pc.turned_pro is null then (select avg_experience from avg_player_characteristics)
                else m.year - pc.turned_pro end as experience,
            case when pc_opp.turned_pro is null then (select avg_experience from avg_player_characteristics)
                else m.year - pc_opp.turned_pro end as opp_experience,
        coalesce(elo.score1,0) as elo_score,
        coalesce(elo.score2,0) as opp_elo_score,
        coalesce(h2h_ml.avg_odds, 0.5) as historical_avg_odds,
        coalesce(ml.avg_odds,0.5) as prev_odds,
        coalesce(ml_opp.avg_odds,0.5) as opp_prev_odds,
        coalesce(ml.fave_wins,0.5) as fave_wins,
        coalesce(ml_opp.fave_wins,0.5) as opp_fave_wins,
        coalesce(ml.fave_losses,0.5) as fave_losses,
        coalesce(ml_opp.fave_losses,0.5) as opp_fave_losses,
        coalesce(ml.under_wins,0.5) as underdog_wins,
        coalesce(ml_opp.under_wins,0.5) as opp_underdog_wins,
        coalesce(ml.under_losses,0.5) as underdog_losses,
        coalesce(ml_opp.under_losses,0.5) as opp_underdog_losses,
        m.year-coalesce(prior_best_year.best_year,m.year) as best_year,
        m.year-coalesce(prior_best_year_opp.best_year,m.year) as opp_best_year,
        m.year-coalesce(prior_worst_year.worst_year,m.year) as worst_year,
        m.year-coalesce(prior_worst_year_opp.worst_year,m.year) as opp_worst_year
        from atp_matches_individual as m
        left outer join atp_matches_prior_h2h as h2h 
            on ((m.player_id,m.opponent_id,m.tournament,m.year)=(h2h.player_id,h2h.opponent_id,h2h.tournament,h2h.year))
        left outer join atp_matches_prior_quarter as prev_quarter
            on ((m.player_id,m.tournament,m.year)=(prev_quarter.player_id,prev_quarter.tournament,prev_quarter.year))
        left outer join atp_matches_prior_quarter as prev_quarter_opp
            on ((m.opponent_id,m.tournament,m.year)=(prev_quarter_opp.player_id,prev_quarter_opp.tournament,prev_quarter_opp.year))
        left outer join atp_matches_prior_year as prev_year
            on ((m.player_id,m.tournament,m.year)=(prev_year.player_id,prev_year.tournament,prev_year.year))
        left outer join atp_matches_tournament_history as tourney_hist
            on ((m.player_id,m.tournament,m.year)=(tourney_hist.player_id,tourney_hist.tournament,tourney_hist.year))
        left outer join atp_matches_prior_year as prev_year_opp
            on ((m.opponent_id,m.tournament,m.year)=(prev_year_opp.player_id,prev_year_opp.tournament,prev_year_opp.year))
        left outer join atp_matches_prior_majors as majors
            on ((m.player_id,m.tournament,m.year)=(majors.player_id,majors.tournament,majors.year))
        left outer join atp_matches_prior_majors as opp_majors
            on ((m.opponent_id,m.tournament,m.year)=(opp_majors.player_id,opp_majors.tournament,opp_majors.year))
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
        left outer join atp_matches_prior_match as prior_match
            on ((m.player_id,m.opponent_id,m.tournament,m.year)=(prior_match.player_id,prior_match.opponent_id,prior_match.tournament,prior_match.year))
        left outer join atp_matches_prior_match as prior_match_opp
            on ((m.opponent_id,m.player_id,m.tournament,m.year)=(prior_match_opp.player_id,prior_match_opp.opponent_id,prior_match_opp.tournament,prior_match_opp.year))
        left outer join atp_player_opponent_score as elo
            on ((m.player_id,m.opponent_id,m.tournament,m.year)=(elo.player_id,elo.opponent_id,elo.tournament,elo.year))
        left outer join atp_matches_prior_h2h_money_lines as h2h_ml
            on ((m.player_id,m.opponent_id,m.tournament,m.year)=(h2h_ml.player_id,h2h_ml.opponent_id,h2h_ml.tournament,h2h_ml.year))
        left outer join atp_matches_prior_money_lines as ml
            on ((m.player_id,m.tournament,m.year)=(ml.player_id,ml.tournament,ml.year))
        left outer join atp_matches_prior_money_lines as ml_opp
            on ((m.opponent_id,m.tournament,m.year)=(ml_opp.player_id,ml_opp.tournament,ml_opp.year))
        left outer join atp_matches_prior_best_year as prior_best_year
            on ((m.player_id,m.year)=(prior_best_year.player_id,prior_best_year.year))
        left outer join atp_matches_prior_best_year as prior_best_year_opp
            on ((m.opponent_id,m.year)=(prior_best_year_opp.player_id,prior_best_year_opp.year))
        left outer join atp_matches_prior_worst_year as prior_worst_year
            on ((m.player_id,m.year)=(prior_worst_year.player_id,prior_worst_year.year))
        left outer join atp_matches_prior_worst_year as prior_worst_year_opp
            on ((m.opponent_id,m.year)=(prior_worst_year_opp.player_id,prior_worst_year_opp.year))
        left outer join atp_matches_qualifying as qualifying
            on ((m.player_id,m.year,m.tournament)=(qualifying.player_id,qualifying.year,qualifying.tournament))
        left outer join atp_matches_qualifying as qualifying_opp
            on ((m.opponent_id,m.year,m.tournament)=(qualifying_opp.player_id,qualifying_opp.year,qualifying_opp.tournament))
        join atp_matches_round as r
            on ((m.player_id,m.opponent_id,m.year,m.tournament)=(r.player_id,r.opponent_id,r.year,r.tournament))
        where m.year <= {{END_DATE}} and m.year >= {{START_DATE}} and not m.round like '%%Qualifying%%' 
    '''.replace('{{END_DATE}}', str(test_season)).replace('{{START_DATE}}', str(start_year))
    if not keep_nulls:
        sql_str = sql_str + '        and m.first_serve_attempted > 0'
    sql = pd.read_sql(sql_str, conn)
    sql = sql[attributes].astype(np.float64, errors='ignore')
    test_data = sql[sql.year == test_season]
    sql = sql[sql.year != test_season]
    print('Data shape:', sql.shape)
    print('Test shape:', test_data.shape)
    return sql, test_data


def get_all_data(all_attributes, test_season=2017, start_year=2003, tournament=None, include_spread=False):
    all_data = load_data(all_attributes, test_season=test_season, start_year=start_year, keep_nulls=tournament is not None)
    data, test_data = all_data
    if tournament is not None:
        data = data[(data.tournament==tournament)]
        test_data = test_data[(test_data.tournament==tournament)]
        print('data size after tournament filter:', data.shape, test_data.shape)
    return data, test_data


# previous year quality
input_attributes0 = [
    'tourney_hist_avg_round',
    'prev_year_avg_round',
    'prev_year_prior_victories',
    #'prev_year_prior_losses',
    'prior_year_match_closeness',

    # prior quarter

    'prior_quarter_games_per_set',
    'prior_quarter_victories',
    # 'prior_quarter_losses',
    # 'prior_quarter_match_closeness',

    # player qualities

    'elo_score',
    'age',
    'surface_experience',
    'height',
    'best_year',

    # match stats

    #'mean_second_serve_points_made',
    #'mean_break_points_made',
    #'mean_break_points_against',
    #'tiebreak_win_percent',
    'major_encounters',

    # previous match
    'previous_games_total',
   # 'duration_prev_match',  DONT HAVE THE DATA FOR CURRENT
   # 'had_qualifier'         DONT HAVE THE DATA FOR CURRENT
]

# opponent attrs
opp_input_attributes0 = ['opp_'+attr for attr in input_attributes0]
input_attributes0 = input_attributes0 + opp_input_attributes0

input_attributes_spread = [
    'prev_year_avg_round',
    'opp_prev_year_avg_round',
    'opp_tourney_hist_avg_round',
    'tourney_hist_avg_round',
    'surface_experience',
    'opp_surface_experience',
    'experience',
    'opp_experience',
    'prior_quarter_avg_round',
    'opp_prior_quarter_avg_round',
    'avg_games_per_set',
    'opp_avg_games_per_set',
    'prior_quarter_match_closeness',
    'opp_prior_quarter_match_closeness',
    'major_encounters',
    'opp_major_encounters',
    'major_avg_round',
    'opp_major_avg_round',
    'major_match_closeness',
    'opp_major_match_closeness',
    'duration_prev_match',
    'opp_duration_prev_match'
]


input_attributes_totals = [
    'mean_second_serve_points_made',
    'opp_mean_second_serve_points_made',
    #'mean_first_serve_points_made',
    #'opp_mean_first_serve_points_made',
    #'prev_year_prior_encounters',
    #'opp_prev_year_prior_encounters',
    'prev_year_avg_round',
    'opp_prev_year_avg_round',
    'surface_experience',
    'opp_surface_experience',
    'elo_score',
    'opp_elo_score',
    'avg_games_per_set',
    'opp_avg_games_per_set',
    'prior_quarter_match_closeness',
    'opp_prior_quarter_match_closeness',
    #'grand_slam',
    'round_num',
    'major_encounters',
    'opp_major_encounters',
    'duration_prev_match',
    'opp_duration_prev_match'
]

y = 'y'
y_spread = 'spread'
y_total_games = 'totals'
y_total_sets = 'num_sets'
y_total_sets_bin = 'num_sets_greater_than_2'

all_attributes = list(input_attributes0)
all_attributes.append(y)
all_attributes.append(y_spread)
all_attributes.append(y_total_games)
all_attributes.append(y_total_sets)
all_attributes.append(y_total_sets_bin)

meta_attributes = ['player_id', 'opponent_id', 'tournament', 'year', 'grand_slam', 'round_num', 'court_surface']
for attr in input_attributes_spread:
    if attr not in all_attributes:
        all_attributes.append(attr)
for attr in opp_input_attributes0:
    if attr not in all_attributes:
        all_attributes.append(attr)
for attr in input_attributes_totals:
    if attr not in all_attributes:
        all_attributes.append(attr)
for meta in meta_attributes:
    if meta not in all_attributes:
        all_attributes.append(meta)

if __name__ == '__main__':
    save = False
    train_spread_model = True
    train_outcome_model = True
    train_total_sets_model = True
    train_total_games_model = False
    sql, test_data = load_data(all_attributes, test_season=2011, start_year=1996)
    sql_slam = sql[sql.grand_slam > 0.5]
    sql = sql[sql.grand_slam < 0.5]
    test_data_slam = test_data[test_data.grand_slam > 0.5]
    test_data = test_data[test_data.grand_slam < 0.5]
    if train_outcome_model:
        model_file = 'tennis_match_outcome_logit.statmodel'
        # print('Attrs: ', sql[all_attributes][0:20])
        # model to predict the total score (h_pts + a_pts)
        # grand slam model
        print('Grand Slam')
        results = smf.logit(y + ' ~ ' + '+'.join(input_attributes0), data=sql_slam).fit()
        print(results.summary())
        _, avg_error = test_model(results, test_data_slam, test_data_slam[y], include_binary=False)
        print('Average error: ', avg_error)
        if save:
            results.save(model_file + '_slam')

        # regular model
        print('Regular model')
        results = smf.logit(y + ' ~ ' + '+'.join(input_attributes0), data=sql).fit()
        print(results.summary())
        _, avg_error = test_model(results, test_data, test_data[y], include_binary=False)
        print('Average error: ', avg_error)
        if save:
            results.save(model_file + '_regular')

    if train_total_games_model:
        model_file = 'tennis_match_totals_logit.statmodel'
        # print('Attrs: ', sql[all_attributes][0:20])
        # model to predict the total score (h_pts + a_pts)
        # grand slam model
        print('Grand Slam')
        results = smf.ols(y_total_games+' ~ '+'+'.join(input_attributes_totals), data=sql_slam).fit()
        print(results.summary())
        _, avg_error = test_model(results, test_data_slam, test_data_slam[y], include_binary=False)
        print('Average error: ', avg_error)
        if save:
            results.save(model_file+'_slam')

        # regular model
        print('Regular model')
        results = smf.ols(y_total_games+' ~ '+'+'.join(input_attributes_totals), data=sql).fit()
        print(results.summary())
        _, avg_error = test_model(results, test_data, test_data[y], include_binary=False)
        print('Average error: ', avg_error)
        if save:
            results.save(model_file+'_regular')

    if train_total_sets_model:
        model_file = 'tennis_match_total_sets_logit.statmodel'
        # print('Attrs: ', sql[all_attributes][0:20])
        # model to predict the total score (h_pts + a_pts)
        # grand slam model
        print('Grand Slam')
        results = smf.ols(y_total_sets+' ~ '+'+'.join(input_attributes_totals), data=sql_slam).fit()
        print(results.summary())
        _, avg_error = test_model(results, test_data_slam, test_data_slam[y], include_binary=False)
        print('Average error: ', avg_error)
        if save:
            results.save(model_file+'_slam')

        # regular model
        print('Regular model')
        results = smf.logit(y_total_sets_bin+' ~ '+'+'.join(input_attributes_totals), data=sql).fit()
        print(results.summary())
        _, avg_error = test_model(results, test_data, test_data[y], include_binary=False)
        print('Average error: ', avg_error)
        if save:
            results.save(model_file+'_regular')

    if train_spread_model:
        model_file = 'tennis_match_spread_logit.statmodel'
        # print('Attrs: ', sql[all_attributes][0:20])
        # model to predict the total score (h_pts + a_pts)
        print('Grand Slam')
        results = smf.ols(y_spread + ' ~ ' + '+'.join(input_attributes_spread), data=sql_slam).fit()
        print(results.summary())
        _, avg_error = test_model(results, test_data_slam, test_data_slam[y], include_binary=False)
        print('Average error: ', avg_error)
        if save:
            results.save(model_file + '_slam')

        # regular model
        print('Regular model')
        results = smf.ols(y_spread + ' ~ ' + '+'.join(input_attributes_spread), data=sql).fit()
        print(results.summary())
        _, avg_error = test_model(results, test_data, test_data[y], include_binary=False)
        print('Average error: ', avg_error)
        if save:
            results.save(model_file + '_regular')

    exit(0)


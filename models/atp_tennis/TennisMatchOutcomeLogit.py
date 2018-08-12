import pandas as pd
from sqlalchemy import create_engine
import statsmodels.formula.api as smf
import numpy as np
import datetime
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


def load_data(attributes, test_season='2017-01-01', start_year='1995-01-01', keep_nulls=False, masters_min=101):
    conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
    sql_str = '''
        select 
            (m.games_won+m.games_against)::double precision as totals,
            (m.games_won+m.games_against)::double precision/18.0 as totals_percent,
            (m.games_won-m.games_against)::double precision as spread,
            (m.games_won-m.games_against)::double precision/6.0 as spread_percent,
            case when m.player_victory is null then null else case when m.player_victory then 1.0 else 0.0 end end as y, 
            case when m.court_surface = 'Clay' then 1.0 else 0.0 end as clay,
            case when m.court_surface = 'Grass' then 1.0 else 0.0 end as grass,
            case when m.court_surface = 'Hard' then 1.0 else 0.0 end as hard,
            m.court_surface as court_surface,
            coalesce(m.aces, 0) as current_aces,
            coalesce(m_opp.aces, 0) as opp_current_aces,
            coalesce(m.double_faults, 0) as current_double_faults,
            coalesce(m_opp.double_faults, 0) as opp_current_double_faults,
            coalesce(m.break_points_made, 0) as current_break_points_made,
            coalesce(m_opp.break_points_made, 0) as opp_current_break_points_made,
            coalesce(m.first_serve_points_made, 0) as current_first_serve_points_made,
            coalesce(m_opp.first_serve_points_made, 0) as opp_current_first_serve_points_made,
            coalesce(m.first_serve_points_attempted, 0) as current_first_serve_points_attempted,
            coalesce(m_opp.first_serve_points_attempted, 0) as opp_current_first_serve_points_attempted,
            coalesce(m.first_serve_made, 0) as current_first_serve_made,
            coalesce(m_opp.first_serve_made, 0) as opp_current_first_serve_made,
            coalesce(m.first_serve_attempted, 0) as current_first_serve_attempted,
            coalesce(m_opp.first_serve_attempted, 0) as opp_first_serve_attempted,            
            coalesce(m.return_points_won, 0) as current_return_points_won,
            coalesce(m_opp.return_points_won, 0) as opp_current_return_points_won,
            m.year as year,
            m.start_date as start_date,
            r.round as round_num,
            r.round::float/7.0 as round_num_percent,
            m.player_id as player_id,
            m.opponent_id as opponent_id,
            m.tournament as tournament,
            m.num_sets as num_sets,
            case when t.masters=100 then 1.0 else 0.0 end as challenger,
            case when t.masters=25 then 1.0 else 0.0 end as itf,
            case when m.num_sets > 2 then 1 else 0 end as num_sets_greater_than_2,
            case when m.tournament in ('roland-garros','wimbledon','us-open','australian-open') or coalesce(greatest(m.num_sets-m.sets_won, m.sets_won)=3,'f')
                then 1.0 else 0.0 end as grand_slam,
            case when coalesce(m.seed,'')='Q' or qualifying.had_qualifier then 1.0 else 0.0 end as had_qualifier,
            case when coalesce(m.seed,'')='WC' then 1.0 else 0.0 end as wild_card,
            case when coalesce(m.seed,'')='PR' then 1.0 else 0.0 end as protected_ranking,
            case when coalesce(m.seed,'asdgas') ~ '^[0-9]+$' and m.seed::integer <= 2 ^ (5 - tournament_first_round.first_round) then 1.0 else 0.0 end as seeded,
            case when coalesce(m_opp.seed,'')='Q' or qualifying_opp.had_qualifier then 1.0 else 0.0 end as opp_had_qualifier,
            case when coalesce(m_opp.seed,'')='WC' then 1.0 else 0.0 end as opp_wild_card,
            case when coalesce(m_opp.seed,'')='PR' then 1.0 else 0.0 end as opp_protected_ranking,
            case when coalesce(m_opp.seed,'asdgas') ~ '^[0-9]+$' and m_opp.seed::integer <= 2 ^ (5 - tournament_first_round.first_round) then 1.0 else 0.0 end as opp_seeded,
            case when coalesce(nation.nationality, pc.country) is null then 0.0 else case when coalesce(nation.nationality,pc.country)=t.location then 1.0 else 0.0 end end as local_player,
            case when coalesce(nation_opp.nationality,pc_opp.country) is null then 0.0 else case when coalesce(nation_opp.nationality,pc_opp.country)=t.location then 1.0 else 0.0 end end as opp_local_player,
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
            coalesce(challengers.prior_encounters,0) as challenger_encounters,
            coalesce(opp_challengers.prior_encounters, 0) as opp_challenger_encounters,
            coalesce(challengers.prior_victories,0) as challenger_victories,
            coalesce(opp_challengers.prior_victories, 0) as opp_challenger_victories,
            coalesce(challengers.avg_round,0) as challenger_avg_round,
            coalesce(opp_challengers.avg_round, 0) as opp_challenger_avg_round,
            coalesce(challengers.avg_games_per_set,0) as challenger_games_per_set,
            coalesce(opp_challengers.avg_games_per_set, 0) as opp_challenger_games_per_set,
            coalesce(challengers.avg_match_closeness,0) as challenger_match_closeness,
            coalesce(opp_challengers.avg_match_closeness, 0) as opp_challenger_match_closeness,
            coalesce(itf.prior_encounters,0) as itf_encounters,
            coalesce(opp_itf.prior_encounters, 0) as opp_itf_encounters,
            coalesce(itf.prior_victories,0) as itf_victories,
            coalesce(opp_itf.prior_victories, 0) as opp_itf_victories,
            coalesce(itf.avg_round,0) as itf_avg_round,
            coalesce(opp_itf.avg_round, 0) as opp_itf_avg_round,
            coalesce(itf.avg_games_per_set,0) as itf_games_per_set,
            coalesce(opp_itf.avg_games_per_set, 0) as opp_itf_games_per_set,
            coalesce(itf.avg_match_closeness,0) as itf_match_closeness,
            coalesce(opp_itf.avg_match_closeness, 0) as opp_itf_match_closeness,
            coalesce(masters.prior_encounters,0) as master_encounters,
            coalesce(opp_masters.prior_encounters, 0) as opp_master_encounters,
            coalesce(masters.prior_victories,0) as master_victories,
            coalesce(opp_masters.prior_victories, 0) as opp_master_victories,
            coalesce(masters.avg_round,0) as master_avg_round,
            coalesce(opp_masters.avg_round, 0) as opp_master_avg_round,
            coalesce(masters.avg_games_per_set,0) as master_games_per_set,
            coalesce(opp_masters.avg_games_per_set, 0) as opp_master_games_per_set,
            coalesce(masters.avg_match_closeness,0) as master_match_closeness,
            coalesce(opp_masters.avg_match_closeness, 0) as opp_master_match_closeness,
            coalesce(matches_250.prior_encounters,0) as encounters_250,
            coalesce(opp_matches_250.prior_encounters, 0) as opp_encounters_250,
            coalesce(matches_250.prior_victories,0) as victories_250,
            coalesce(opp_matches_250.prior_victories, 0) as opp_victories_250,
            coalesce(matches_250.avg_round,0) as avg_round_250,
            coalesce(opp_matches_250.avg_round, 0) as opp_avg_round_250,
            coalesce(matches_250.avg_games_per_set,0) as games_per_set_250,
            coalesce(opp_matches_250.avg_games_per_set, 0) as opp_games_per_set_250,
            coalesce(matches_250.avg_match_closeness,0) as match_closeness_250,
            coalesce(opp_matches_250.avg_match_closeness, 0) as opp_match_closeness_250,
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
            coalesce(prev_year.victory_closeness,0) as prev_year_victory_closeness,
            coalesce(prev_year_opp.victory_closeness,0) as opp_prev_year_victory_closeness,
            coalesce(prev_year.loss_closeness,0) as prev_year_loss_closeness,
            coalesce(prev_year_opp.loss_closeness,0) as opp_prev_year_loss_closeness,
            coalesce(prev_year.match_recovery::float/greatest(prev_year.prior_encounters,1),0) as prior_year_match_recovery,
            coalesce(prev_year_opp.match_recovery::float/greatest(prev_year_opp.prior_encounters,1),0) as opp_prior_year_match_recovery,
            coalesce(prev_quarter.match_recovery::float/greatest(prev_year.prior_encounters,1),0) as prior_quarter_match_recovery,
            coalesce(prev_quarter_opp.match_recovery::float/greatest(prev_year_opp.prior_encounters,1),0) as opp_prior_quarter_match_recovery,
            coalesce(prev_year.match_collapse,0) as prior_year_match_collapse,
            coalesce(prev_year_opp.match_collapse,0) as opp_prior_year_match_collapse,
            coalesce(prev_2year.avg_match_closeness,0) as prior_2year_match_closeness,
            coalesce(prev_2year_opp.avg_match_closeness,0) as opp_prior_2year_match_closeness,
            coalesce(prev_2year.avg_round,0) as prev_2year_avg_round,
            coalesce(prev_2year_opp.avg_round,0) as opp_prev_2year_avg_round,
            coalesce(prev_2year.match_recovery::float/greatest(prev_2year.prior_encounters,1),0) as prior_2year_match_recovery,
            coalesce(prev_2year_opp.match_recovery::float/greatest(prev_2year_opp.prior_encounters,1),0) as opp_prior_2year_match_recovery,
            coalesce(prev_2year.match_collapse,0) as prior_2year_match_collapse,
            coalesce(prev_2year_opp.match_collapse,0) as opp_prior_2year_match_collapse,
            coalesce(prev_quarter.avg_match_closeness,0) as prior_quarter_match_closeness,
            coalesce(prev_quarter_opp.avg_match_closeness,0) as opp_prior_quarter_match_closeness,
            coalesce(prev_quarter.avg_games_per_set,0) as prior_quarter_games_per_set,
            coalesce(prev_quarter_opp.avg_games_per_set,0) as opp_prior_quarter_games_per_set,
            coalesce(prev_quarter.prior_victories,0) as prior_quarter_victories,
            coalesce(prev_quarter_opp.prior_victories,0) as opp_prior_quarter_victories,
            coalesce(prev_quarter.prior_losses,0) as prior_quarter_losses,
            coalesce(prev_quarter_opp.prior_losses,0) as opp_prior_quarter_losses,
            coalesce(prev_2year.avg_games_per_set,0) as prior_2year_games_per_set,
            coalesce(prev_2year_opp.avg_games_per_set,0) as opp_prior_2year_games_per_set,
            coalesce(prev_2year.prior_victories,0) as prior_2year_victories,
            coalesce(prev_2year_opp.prior_victories,0) as opp_prior_2year_victories,
            coalesce(prev_2year.prior_losses,0) as prior_2year_losses,
            coalesce(prev_2year_opp.prior_losses,0) as opp_prior_2year_losses,
            coalesce(prev_2year.prior_encounters,0) as prev_2year_prior_encounters,
            coalesce(prev_2year_opp.prior_encounters,0) as opp_prev_2year_prior_encounters,
            coalesce(prev_2year.prior_victories::float/greatest(1,prev_2year.prior_encounters),0) as prev_2year_win_percent,
            coalesce(prev_2year_opp.prior_victories::float/greatest(1,prev_2year_opp.prior_encounters),0) as opp_prev_2year_win_percent,
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
            coalesce(mean.aces,0) as mean_aces,
            coalesce(mean_opp.aces,0) as opp_mean_aces,
            coalesce(mean.double_faults,0.5) as mean_faults,
            coalesce(mean_opp.double_faults,0.5) as opp_mean_faults,
            coalesce(mean.service_points_won,1) as mean_service_points_won,
            coalesce(mean_opp.service_points_won,1) as opp_mean_service_points_won,
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
            coalesce(se.tiebreaks_won,0)::float/(1+coalesce(se.tiebreaks_total,0)) as tiebreak_win_percent,
            coalesce(se_opp.tiebreaks_won,0)::float/(1+coalesce(se_opp.tiebreaks_total,0)) as opp_tiebreak_win_percent,
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
            extract(epoch from coalesce(prior_match.duration,'01:30:00'::time))::float/3600.0 as duration_prev_match,
            extract(epoch from coalesce(prior_match_opp.duration,'01:30:00'::time))::float/3600.0 as opp_duration_prev_match,         
            coalesce(prior_match.games_won, 6) + coalesce(prior_match.games_against, 6) as previous_games_total,
            coalesce(prior_match_opp.games_won, 6) + coalesce(prior_match_opp.games_against, 6) as opp_previous_games_total,
            coalesce(prior_matches.games_won, 6) + coalesce(prior_matches.games_against, 0) as previous_games_total2,
            coalesce(prior_matches_opp.games_won, 6) + coalesce(prior_matches_opp.games_against, 0) as opp_previous_games_total2,
            extract(year from m.start_date) - extract(year from pc.date_of_birth) as age,
            extract(year from m.start_date) - extract(year from pc_opp.date_of_birth) as opp_age,
            case when pc.turned_pro is null then (select avg_experience from avg_player_characteristics)
                else extract(year from m.start_date) - pc.turned_pro end as experience,
            case when pc_opp.turned_pro is null then (select avg_experience from avg_player_characteristics)
                else extract(year from m.start_date) - pc_opp.turned_pro end as opp_experience,
        coalesce(elo.score1,100) as elo_score,
        coalesce(elo.score2,100) as opp_elo_score,
        coalesce(elo.weighted_score1,100) as elo_score_weighted,
        coalesce(elo.weighted_score2,100) as opp_elo_score_weighted,
        coalesce(atp_rank.score,0) as atp_rank,
        coalesce(atp_rank_opp.score,0) as opp_atp_rank,
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
        m.year-coalesce(prior_worst_year_opp.worst_year,m.year) as opp_worst_year,
        coalesce(se.num_injuries, 0) as injuries,
        coalesce(se_opp.num_injuries, 0) as opp_injuries,
        coalesce(se.last_tournament_time, 365.25*4)::float/(365.25*4) as last_tournament_time,
        coalesce(se_opp.last_tournament_time, 365.25*4)::float/(365.25*4) as opp_last_tournament_time,
        coalesce(se.last_itf_tournament_time, 365.25*4)::float/(365.25*4) as last_itf_tournament_time,
        coalesce(se_opp.last_itf_tournament_time, 365.25*4)::float/(365.25*4) as opp_last_itf_tournament_time,
        coalesce(se.last_challenger_tournament_time, 365.25*4)::float/(365.25*4) as last_challenger_tournament_time,
        coalesce(se_opp.last_challenger_tournament_time, 365.25*4)::float/(365.25*4) as opp_last_challenger_tournament_time,
        coalesce(se.last_atp_tournament_time, 365.25*4)::float/(365.25*4) as last_atp_tournament_time,
        coalesce(se_opp.last_atp_tournament_time, 365.25*4)::float/(365.25*4) as opp_last_atp_tournament_time,
        coalesce(se.months_not_played, 0) as not_played,
        coalesce(se_opp.months_not_played, 0) as opp_not_played,
        coalesce(se.percent_itf, 0) as percent_itf,
        coalesce(se_opp.percent_itf, 0) as opp_percent_itf,
        coalesce(se.percent_challenger, 0) as percent_challenger,
        coalesce(se_opp.percent_challenger, 0) as opp_percent_challenger,
        coalesce(se.percent_majors, 0) as percent_majors,
        coalesce(se_opp.percent_majors, 0) as opp_percent_majors,                                
        coalesce(se.tournaments_per_year, 0) as tournaments_per_year,
        coalesce(se_opp.tournaments_per_year, 0) as opp_tournaments_per_year,     
        coalesce(t.masters, 250) as tournament_rank,
        coalesce(t.masters, 250)::double precision/2000 as tournament_rank_percent,
        (m.start_date - first_match.first_date)::float/365.25 as first_match_date,
        (m.start_date - first_match_opp.first_date)::float/365.25 as opp_first_match_date,
        case when r.round=tournament_first_round.first_round then 1.0 else 0.0 end as first_round,
        case when coalesce(dubs.played_doubles, 'f') then 1.0 else 0.0 end as played_doubles,
        case when coalesce(dubs_opp.played_doubles, 'f') then 1.0 else 0.0 end as opp_played_doubles
        from atp_matches_individual as m
        join atp_matches_individual as m_opp
        on ((m.opponent_id,m.player_id,m.tournament,m.start_date)=(m_opp.player_id,m_opp.opponent_id,m_opp.tournament,m_opp.start_date))
        join atp_matches_round as r
            on ((m.player_id,m.opponent_id,m.start_date,m.tournament)=(r.player_id,r.opponent_id,r.start_date,r.tournament))
        join atp_tournament_first_round as tournament_first_round on ((tournament_first_round.tournament,tournament_first_round.start_date)=(m.tournament,m.start_date))
        join atp_tournament_dates as t on ((m.start_date,m.tournament)=(t.start_date,t.tournament))
        left outer join atp_matches_prior_h2h as h2h 
            on ((m.player_id,m.opponent_id,m.tournament,m.start_date)=(h2h.player_id,h2h.opponent_id,h2h.tournament,h2h.start_date))
        left outer join atp_player_nationality as nation
            on (m.player_id=nation.player_id)
        left outer join atp_player_nationality as nation_opp
            on (m.opponent_id=nation_opp.player_id)
        left outer join atp_matches_prior_quarter as prev_quarter
            on ((m.player_id,m.tournament,m.start_date)=(prev_quarter.player_id,prev_quarter.tournament,prev_quarter.start_date))
        left outer join atp_matches_prior_quarter as prev_quarter_opp
            on ((m.opponent_id,m.tournament,m.start_date)=(prev_quarter_opp.player_id,prev_quarter_opp.tournament,prev_quarter_opp.start_date))
        left outer join atp_matches_prior_year as prev_year
            on ((m.player_id,m.tournament,m.start_date)=(prev_year.player_id,prev_year.tournament,prev_year.start_date))
        left outer join atp_matches_prior_year as prev_year_opp
            on ((m.opponent_id,m.tournament,m.start_date)=(prev_year_opp.player_id,prev_year_opp.tournament,prev_year_opp.start_date))
        left outer join atp_matches_prior_2years as prev_2year
            on ((m.player_id,m.tournament,m.start_date)=(prev_2year.player_id,prev_2year.tournament,prev_2year.start_date))
        left outer join atp_matches_prior_2years as prev_2year_opp
            on ((m.opponent_id,m.tournament,m.start_date)=(prev_2year_opp.player_id,prev_2year_opp.tournament,prev_2year_opp.start_date))
        left outer join atp_matches_tournament_history as tourney_hist
            on ((m.player_id,m.tournament,m.start_date)=(tourney_hist.player_id,tourney_hist.tournament,tourney_hist.start_date))        
        left outer join atp_matches_prior_majors as majors
            on ((m.player_id,m.tournament,m.start_date)=(majors.player_id,majors.tournament,majors.start_date))
        left outer join atp_matches_prior_majors as opp_majors
            on ((m.opponent_id,m.tournament,m.start_date)=(opp_majors.player_id,opp_majors.tournament,opp_majors.start_date))
        left outer join atp_matches_prior_challenger as challengers
            on ((m.player_id,m.tournament,m.start_date)=(challengers.player_id,challengers.tournament,challengers.start_date))
        left outer join atp_matches_prior_challenger as opp_challengers
            on ((m.opponent_id,m.tournament,m.start_date)=(opp_challengers.player_id,opp_challengers.tournament,opp_challengers.start_date))
        left outer join atp_matches_prior_itf as itf
            on ((m.player_id,m.tournament,m.start_date)=(itf.player_id,itf.tournament,itf.start_date))
        left outer join atp_matches_prior_itf as opp_itf
            on ((m.opponent_id,m.tournament,m.start_date)=(opp_itf.player_id,opp_itf.tournament,opp_itf.start_date))
        left outer join atp_matches_prior_masters as masters
            on ((m.player_id,m.tournament,m.start_date)=(masters.player_id,masters.tournament,masters.start_date))
        left outer join atp_matches_prior_masters as opp_masters
            on ((m.opponent_id,m.tournament,m.start_date)=(opp_masters.player_id,opp_masters.tournament,opp_masters.start_date))
        left outer join atp_matches_prior_250 as matches_250
            on ((m.player_id,m.tournament,m.start_date)=(matches_250.player_id,matches_250.tournament,matches_250.start_date))
        left outer join atp_matches_prior_250 as opp_matches_250
            on ((m.opponent_id,m.tournament,m.start_date)=(opp_matches_250.player_id,opp_matches_250.tournament,opp_matches_250.start_date))
        left outer join atp_matches_tournament_history as tourney_hist_opp
            on ((m.opponent_id,m.tournament,m.start_date)=(tourney_hist_opp.player_id,tourney_hist_opp.tournament,tourney_hist_opp.start_date))
        left outer join atp_matches_prior_year_avg as mean
            on ((m.player_id,m.tournament,m.start_date)=(mean.player_id,mean.tournament,mean.start_date))
        left outer join atp_matches_prior_year_avg as mean_opp
            on ((m.opponent_id,m.tournament,m.start_date)=(mean_opp.player_id,mean_opp.tournament,mean_opp.start_date))   
        left outer join atp_matches_prior_year_tournament_round as prior_tourney
            on ((m.player_id,m.tournament,m.start_date)=(prior_tourney.player_id,prior_tourney.tournament,prior_tourney.start_date))
        left outer join atp_matches_prior_year_tournament_round as prior_tourney_opp
            on ((m.opponent_id,m.tournament,m.start_date)=(prior_tourney_opp.player_id,prior_tourney_opp.tournament,prior_tourney_opp.start_date)) 
        left outer join atp_matches_prior_experience as se 
            on ((m.player_id,m.tournament,m.start_date)=(se.player_id,se.tournament,se.start_date))
        left outer join atp_matches_prior_experience as se_opp
            on ((m.opponent_id,m.tournament,m.start_date)=(se_opp.player_id,se_opp.tournament,se_opp.start_date))
        left outer join atp_player_characteristics as pc
            on ((m.player_id,m.tournament,m.start_date)=(pc.player_id,pc.tournament,pc.start_date))
        left outer join atp_player_characteristics as pc_opp
            on ((m.opponent_id,m.tournament,m.start_date)=(pc_opp.player_id,pc_opp.tournament,pc_opp.start_date))
        left outer join atp_matches_prior_match as prior_match
            on ((m.player_id,m.opponent_id,m.tournament,m.start_date)=(prior_match.player_id,prior_match.opponent_id,prior_match.tournament,prior_match.start_date))
        left outer join atp_matches_prior_match as prior_match_opp
            on ((m.opponent_id,m.player_id,m.tournament,m.start_date)=(prior_match_opp.player_id,prior_match_opp.opponent_id,prior_match_opp.tournament,prior_match_opp.start_date))
        left outer join atp_matches_prior_matches as prior_matches
            on ((m.player_id,m.opponent_id,m.tournament,m.start_date)=(prior_matches.player_id,prior_matches.opponent_id,prior_matches.tournament,prior_matches.start_date))
        left outer join atp_matches_prior_matches as prior_matches_opp
            on ((m.opponent_id,m.player_id,m.tournament,m.start_date)=(prior_matches_opp.player_id,prior_matches_opp.opponent_id,prior_matches_opp.tournament,prior_matches_opp.start_date))
        left outer join atp_player_opponent_score as elo
            on ((m.player_id,m.opponent_id,m.tournament,m.start_date)=(elo.player_id,elo.opponent_id,elo.tournament,elo.start_date))
        left outer join atp_matches_prior_h2h_money_lines as h2h_ml
            on ((m.player_id,m.opponent_id,m.tournament,m.start_date)=(h2h_ml.player_id,h2h_ml.opponent_id,h2h_ml.tournament,h2h_ml.start_date))
        left outer join atp_matches_prior_money_lines as ml
            on ((m.player_id,m.tournament,m.start_date)=(ml.player_id,ml.tournament,ml.start_date))
        left outer join atp_matches_prior_money_lines as ml_opp
            on ((m.opponent_id,m.tournament,m.start_date)=(ml_opp.player_id,ml_opp.tournament,ml_opp.start_date))
        left outer join atp_matches_prior_best_year as prior_best_year
            on ((m.player_id,m.start_date)=(prior_best_year.player_id,prior_best_year.start_date))
        left outer join atp_matches_prior_best_year as prior_best_year_opp
            on ((m.opponent_id,m.start_date)=(prior_best_year_opp.player_id,prior_best_year_opp.start_date))
        left outer join atp_matches_prior_worst_year as prior_worst_year
            on ((m.player_id,m.start_date)=(prior_worst_year.player_id,prior_worst_year.start_date))
        left outer join atp_matches_prior_worst_year as prior_worst_year_opp
            on ((m.opponent_id,m.start_date)=(prior_worst_year_opp.player_id,prior_worst_year_opp.start_date))
        left outer join atp_matches_qualifying as qualifying
            on ((m.player_id,m.start_date,m.tournament)=(qualifying.player_id,qualifying.start_date,qualifying.tournament))
        left outer join atp_matches_qualifying as qualifying_opp
            on ((m.opponent_id,m.start_date,m.tournament)=(qualifying_opp.player_id,qualifying_opp.start_date,qualifying_opp.tournament))
        left outer join atp_players_first_match as first_match
            on (m.player_id=first_match.player_id)
        left outer join atp_players_first_match as first_match_opp
            on (m.opponent_id=first_match_opp.player_id)
        left outer join atp_matches_played_doubles as dubs
            on ((m.player_id,m.start_date,m.tournament)=(dubs.player_id,dubs.start_date,dubs.tournament))
        left outer join atp_matches_played_doubles as dubs_opp
            on ((m.opponent_id,m.start_date,m.tournament)=(dubs_opp.player_id,dubs_opp.start_date,dubs_opp.tournament))
        left outer join atp_player_atp_rank as atp_rank
            on ((m.player_id,m.start_date,m.tournament)=(atp_rank.player_id,atp_rank.start_date,atp_rank.tournament))
        left outer join atp_player_atp_rank as atp_rank_opp
            on ((m.opponent_id,m.start_date,m.tournament)=(atp_rank_opp.player_id,atp_rank_opp.start_date,atp_rank_opp.tournament))
        where t.masters > {{MASTERS_MIN}} and not m.doubles and m.court_surface in ('Clay', 'Hard', 'Grass') and prev_year.prior_encounters is not null and prev_year.prior_encounters > 0 and prev_year_opp.prior_encounters is not null and prev_year_opp.prior_encounters > 0 and tournament_first_round.first_round is not null and (m.retirement is null or not m.retirement) and r.round is not null and r.round > 0 and m.start_date < '{{END_DATE}}'::date and m.start_date >= '{{START_DATE}}'::date and not m.round like '%%Qualifying%%' 
    '''.replace('{{MASTERS_MIN}}', str(masters_min)).replace('{{END_DATE}}', str(test_season)).replace('{{START_DATE}}', str(start_year))
    if not keep_nulls:
        sql_str = sql_str + '        and m.retirement is not null '
    sql = pd.read_sql(sql_str, conn)
    sql = sql[attributes].astype(np.float64, errors='ignore')
    print('Data shape:', sql.shape)
    return sql


def get_all_data(all_attributes, test_season='2017-01-01', num_test_years=1, start_year='2005-01-01', tournament=None, masters_min=101, num_test_months=0):
    month = int(test_season[5:7])
    year = int(test_season[0:4])
    day = int(test_season[8:10])

    if num_test_months > 0:
        month = month - num_test_months
    while month <= 0:
        year = year - 1
        month = month + 12
    while month > 12:
        year = year + 1
        month = month - 12
    # handle days for months with less than 31 days
    if month == 2 and day > 28:
        day = 28
    elif month in [4, 6, 9, 11] and day > 30:
        day = 30
    date = datetime.date(year-num_test_years, month, day).strftime('%Y-%m-%d')
    data = load_data(all_attributes, test_season=date, start_year=start_year, keep_nulls=False, masters_min=masters_min)
    test_data = load_data(all_attributes, test_season=test_season, start_year=date, keep_nulls=tournament is not None, masters_min=masters_min)
    if tournament is not None:
        #data = data[data.tournament==tournament]
        test_data = test_data[test_data.tournament==tournament]
        print('data size after tournament filter:', data.shape, test_data.shape)
    return data, test_data


# previous year quality
input_attributes0 = [
    'h2h_prior_win_percent',
    'tourney_hist_avg_round',

    'tournaments_per_year',

    'percent_itf',
    'percent_challenger',
    'percent_majors',

    'not_played',
    'last_tournament_time',

    # prior quarter
    'prior_quarter_victories',
    'prior_quarter_losses',

    # prior year
    'prev_year_prior_victories',
    'prev_year_prior_losses',
    'prev_year_avg_round',
    'prev_year_victory_closeness',
    'prev_year_loss_closeness',

    # prior 2 years
    'prior_2year_victories',
    'prior_2year_losses',
    'prev_2year_avg_round',
    'prior_2year_match_recovery',

    # player qualities
    'elo_score_weighted',
    'surface_experience',
    'injuries',
    'first_match_date',
    'best_year',
    'atp_rank',
    'played_doubles',

    'last_itf_tournament_time',
    'last_challenger_tournament_time',
    'last_atp_tournament_time',

    # majors
    'major_avg_round',
    'major_encounters',

    # masters
    'master_avg_round',
    'master_victories',

    # 250 level
    'avg_round_250',
    'encounters_250',

    # challenger
    'challenger_avg_round',
    'challenger_encounters',

    # itf
    'itf_avg_round',
    'itf_encounters',

    'tiebreak_win_percent',

    # current tournament
    'previous_games_total2',
    'had_qualifier',
    'seeded',
    'local_player',
    'wild_card',
]

# opponent attrs
opp_input_attributes0 = ['opp_'+attr for attr in input_attributes0]
opp_input_attributes0.remove('opp_h2h_prior_win_percent')
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

meta_attributes = ['itf', 'challenger', 'start_date', 'first_round', 'tournament_rank', 'clay', 'hard', 'grass', 'prev_year_prior_encounters', 'opp_prev_year_prior_encounters', 'player_id', 'opponent_id', 'tournament', 'year', 'grand_slam', 'round_num', 'court_surface']
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
    train_spread_model = False
    train_outcome_model = True
    train_total_sets_model = False
    train_total_games_model = False
    sql = load_data(all_attributes, test_season='2011-01-01', start_year='1995-01-01')
    test_data = load_data(all_attributes, test_season='2012-01-01', start_year='2011-01-01')
    sql_slam = sql[sql.itf < 0.5]
    sql = sql[sql.itf > 0.5]
    test_data_slam = test_data[test_data.itf < 0.5]
    test_data = test_data[test_data.itf > 0.5]
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


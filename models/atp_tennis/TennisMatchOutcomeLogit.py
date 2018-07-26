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


def load_data(attributes, test_season='2017-01-01', start_year='1995-01-01', keep_nulls=False):
    conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
    sql_str = '''
        select 
            (m.games_won+m.games_against)::double precision as totals,
            (m.games_won-m.games_against)::double precision as spread,
            case when m.player_victory is null then null else case when m.player_victory then 1.0 else 0.0 end end as y, 
            case when m.court_surface = 'Clay' then 1.0 else 0.0 end as clay,
            case when m.court_surface = 'Grass' then 1.0 else 0.0 end as grass,
            case when m.court_surface = 'Hard' then 1.0 else 0.0 end as hard,
            m.court_surface as court_surface,
            m.year as year,
            m.start_date as start_date,
            r.round as round_num,
            m.player_id as player_id,
            m.opponent_id as opponent_id,
            m.tournament as tournament,
            m.num_sets as num_sets,
            case when m.challenger then 1.0 else 0.0 end as challenger,
            case when m.num_sets > 2 then 1 else 0 end as num_sets_greater_than_2,
            case when m.tournament in ('roland-garros','wimbledon','us-open','australian-open') or coalesce(greatest(m.num_sets-m.sets_won, m.sets_won)=3,'f')
                then 1.0 else 0.0 end as grand_slam,
            case when coalesce(m.seed,'')='Q' then 1.0 else 0.0 end as had_qualifier,
            case when coalesce(m.seed,'')='WC' then 1.0 else 0.0 end as wild_card,
            case when coalesce(m.seed,'')='PR' then 1.0 else 0.0 end as protected_ranking,
            case when coalesce(m.seed,'asdgas') ~ '^[0-9]+$' and m.seed::integer <= 2 ^ (5 - tournament_first_round.first_round) then 1.0 else 0.0 end as seeded,
            case when coalesce(m_opp.seed,'')='Q' then 1.0 else 0.0 end as opp_had_qualifier,
            case when coalesce(m_opp.seed,'')='WC' then 1.0 else 0.0 end as opp_wild_card,
            case when coalesce(m_opp.seed,'')='PR' then 1.0 else 0.0 end as opp_protected_ranking,
            case when coalesce(m_opp.seed,'asdgas') ~ '^[0-9]+$' and m_opp.seed::integer <= 2 ^ (5 - tournament_first_round.first_round) then 1.0 else 0.0 end as opp_seeded,
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
            coalesce(matches_250.avg_round,0) as master_avg_round_250,
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
        coalesce(inj.num_injuries, 0) as injuries,
        coalesce(inj_opp.num_injuries, 0) as opp_injuries,
        coalesce(t.masters, 250) as tournament_rank,
        (m.start_date - first_match.first_date)::float/365.25 as first_match_date,
        (m.start_date - first_match_opp.first_date)::float/365.25 as opp_first_match_date,
        case when r.round=tournament_first_round.first_round then 1.0 else 0.0 end as first_round
        from atp_matches_individual as m
        join atp_matches_individual as m_opp
        on ((m.opponent_id,m.player_id,m.tournament,m.start_date,m.challenger)=(m_opp.player_id,m_opp.opponent_id,m_opp.tournament,m_opp.start_date,m_opp.challenger))
        join atp_matches_round as r
            on ((m.player_id,m.opponent_id,m.start_date,m.tournament,m.challenger)=(r.player_id,r.opponent_id,r.start_date,r.tournament, r.challenger))
        join atp_tournament_first_round as tournament_first_round on ((tournament_first_round.tournament,tournament_first_round.start_date,tournament_first_round.challenger)=(m.tournament,m.start_date,m.challenger))
        join atp_tournament_dates as t on ((m.start_date,m.tournament,m.challenger)=(t.start_date,t.tournament,t.challenger))
        left outer join atp_matches_prior_h2h as h2h 
            on ((m.player_id,m.opponent_id,m.tournament,m.start_date,m.challenger)=(h2h.player_id,h2h.opponent_id,h2h.tournament,h2h.start_date,h2h.challenger))
        left outer join atp_matches_prior_quarter as prev_quarter
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(prev_quarter.player_id,prev_quarter.tournament,prev_quarter.start_date,prev_quarter.challenger))
        left outer join atp_matches_prior_quarter as prev_quarter_opp
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(prev_quarter_opp.player_id,prev_quarter_opp.tournament,prev_quarter_opp.start_date,prev_quarter_opp.challenger))
        left outer join atp_matches_prior_year as prev_year
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(prev_year.player_id,prev_year.tournament,prev_year.start_date,prev_year.challenger))
        left outer join atp_matches_prior_year as prev_year_opp
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(prev_year_opp.player_id,prev_year_opp.tournament,prev_year_opp.start_date,prev_year_opp.challenger))
        left outer join atp_matches_prior_2years as prev_2year
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(prev_2year.player_id,prev_2year.tournament,prev_2year.start_date,prev_2year.challenger))
        left outer join atp_matches_prior_2years as prev_2year_opp
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(prev_2year_opp.player_id,prev_2year_opp.tournament,prev_2year_opp.start_date,prev_2year_opp.challenger))
        left outer join atp_matches_tournament_history as tourney_hist
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(tourney_hist.player_id,tourney_hist.tournament,tourney_hist.start_date,tourney_hist.challenger))        
        left outer join atp_matches_prior_majors as majors
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(majors.player_id,majors.tournament,majors.start_date,majors.challenger))
        left outer join atp_matches_prior_majors as opp_majors
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(opp_majors.player_id,opp_majors.tournament,opp_majors.start_date,opp_majors.challenger))
        left outer join atp_matches_prior_challenger as challengers
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(challengers.player_id,challengers.tournament,challengers.start_date,challengers.challenger))
        left outer join atp_matches_prior_challenger as opp_challengers
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(opp_challengers.player_id,opp_challengers.tournament,opp_challengers.start_date,opp_challengers.challenger))
        left outer join atp_matches_prior_masters as masters
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(masters.player_id,masters.tournament,masters.start_date,masters.challenger))
        left outer join atp_matches_prior_masters as opp_masters
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(opp_masters.player_id,opp_masters.tournament,opp_masters.start_date,opp_masters.challenger))
        left outer join atp_matches_prior_250 as matches_250
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(matches_250.player_id,matches_250.tournament,matches_250.start_date,matches_250.challenger))
        left outer join atp_matches_prior_250 as opp_matches_250
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(opp_matches_250.player_id,opp_matches_250.tournament,opp_matches_250.start_date,opp_matches_250.challenger))
        left outer join atp_matches_tournament_history as tourney_hist_opp
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(tourney_hist_opp.player_id,tourney_hist_opp.tournament,tourney_hist_opp.start_date,tourney_hist_opp.challenger))
        left outer join atp_matches_prior_year_avg as mean
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(mean.player_id,mean.tournament,mean.start_date,mean.challenger))
        left outer join atp_matches_prior_year_avg as mean_opp
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(mean_opp.player_id,mean_opp.tournament,mean_opp.start_date,mean_opp.challenger))   
        left outer join atp_matches_prior_year_tournament_round as prior_tourney
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(prior_tourney.player_id,prior_tourney.tournament,prior_tourney.start_date,prior_tourney.challenger))
        left outer join atp_matches_prior_year_tournament_round as prior_tourney_opp
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(prior_tourney_opp.player_id,prior_tourney_opp.tournament,prior_tourney_opp.start_date,prior_tourney_opp.challenger)) 
        left outer join atp_matches_prior_tiebreak_percentage as tb
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(tb.player_id,tb.tournament,tb.start_date,tb.challenger)) 
        left outer join atp_matches_prior_tiebreak_percentage as tb_opp
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(tb_opp.player_id,tb_opp.tournament,tb_opp.start_date,tb_opp.challenger)) 
        left outer join atp_matches_prior_surface_experience as se 
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(se.player_id,se.tournament,se.start_date,se.challenger))
        left outer join atp_matches_prior_surface_experience as se_opp
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(se_opp.player_id,se_opp.tournament,se_opp.start_date,se_opp.challenger))
        left outer join atp_player_characteristics as pc
            on ((m.player_id,m.tournament,m.start_date)=(pc.player_id,pc.tournament,pc.start_date))
        left outer join atp_player_characteristics as pc_opp
            on ((m.opponent_id,m.tournament,m.start_date)=(pc_opp.player_id,pc_opp.tournament,pc_opp.start_date))
        left outer join atp_matches_prior_match as prior_match
            on ((m.player_id,m.opponent_id,m.tournament,m.start_date,m.challenger)=(prior_match.player_id,prior_match.opponent_id,prior_match.tournament,prior_match.start_date,prior_match.challenger))
        left outer join atp_matches_prior_match as prior_match_opp
            on ((m.opponent_id,m.player_id,m.tournament,m.start_date,m.challenger)=(prior_match_opp.player_id,prior_match_opp.opponent_id,prior_match_opp.tournament,prior_match_opp.start_date,prior_match_opp.challenger))
        left outer join atp_matches_prior_matches as prior_matches
            on ((m.player_id,m.opponent_id,m.tournament,m.start_date,m.challenger)=(prior_matches.player_id,prior_matches.opponent_id,prior_matches.tournament,prior_matches.start_date,prior_matches.challenger))
        left outer join atp_matches_prior_matches as prior_matches_opp
            on ((m.opponent_id,m.player_id,m.tournament,m.start_date,m.challenger)=(prior_matches_opp.player_id,prior_matches_opp.opponent_id,prior_matches_opp.tournament,prior_matches_opp.start_date,prior_matches_opp.challenger))
        left outer join atp_player_opponent_score as elo
            on ((m.player_id,m.opponent_id,m.tournament,m.start_date,m.challenger)=(elo.player_id,elo.opponent_id,elo.tournament,elo.start_date,elo.challenger))
        left outer join atp_matches_prior_h2h_money_lines as h2h_ml
            on ((m.player_id,m.opponent_id,m.tournament,m.start_date,m.challenger)=(h2h_ml.player_id,h2h_ml.opponent_id,h2h_ml.tournament,h2h_ml.start_date, h2h_ml.challenger))
        left outer join atp_matches_prior_money_lines as ml
            on ((m.player_id,m.tournament,m.start_date,m.challenger)=(ml.player_id,ml.tournament,ml.start_date,ml.challenger))
        left outer join atp_matches_prior_money_lines as ml_opp
            on ((m.opponent_id,m.tournament,m.start_date,m.challenger)=(ml_opp.player_id,ml_opp.tournament,ml_opp.start_date,ml_opp.challenger))
        left outer join atp_matches_prior_best_year as prior_best_year
            on ((m.player_id,m.start_date,m.challenger)=(prior_best_year.player_id,prior_best_year.start_date,prior_best_year.challenger))
        left outer join atp_matches_prior_best_year as prior_best_year_opp
            on ((m.opponent_id,m.start_date,m.challenger)=(prior_best_year_opp.player_id,prior_best_year_opp.start_date,prior_best_year_opp.challenger))
        left outer join atp_matches_prior_worst_year as prior_worst_year
            on ((m.player_id,m.start_date,m.challenger)=(prior_worst_year.player_id,prior_worst_year.start_date,prior_worst_year.challenger))
        left outer join atp_matches_prior_worst_year as prior_worst_year_opp
            on ((m.opponent_id,m.start_date,m.challenger)=(prior_worst_year_opp.player_id,prior_worst_year_opp.start_date,prior_worst_year_opp.challenger))
        left outer join atp_matches_qualifying as qualifying
            on ((m.player_id,m.start_date,m.tournament,m.challenger)=(qualifying.player_id,qualifying.start_date,qualifying.tournament,qualifying.challenger))
        left outer join atp_matches_qualifying as qualifying_opp
            on ((m.opponent_id,m.start_date,m.tournament,m.challenger)=(qualifying_opp.player_id,qualifying_opp.start_date,qualifying_opp.tournament,qualifying_opp.challenger))
        left outer join atp_matches_injuries as inj
            on ((m.player_id,m.start_date,m.tournament,m.challenger)=(inj.player_id,inj.start_date,inj.tournament,inj.challenger))
        left outer join atp_matches_injuries as inj_opp
            on ((m.opponent_id,m.start_date,m.tournament,m.challenger)=(inj_opp.player_id,inj_opp.start_date,inj_opp.tournament, inj_opp.challenger))
        left outer join atp_players_first_match as first_match
            on (m.player_id=first_match.player_id)
        left outer join atp_players_first_match as first_match_opp
            on (m.opponent_id=first_match_opp.player_id)
        where not m.challenger and prev_year.prior_encounters is not null and prev_year.prior_encounters > 0 and prev_year_opp.prior_encounters is not null and prev_year_opp.prior_encounters > 0 and tournament_first_round.first_round is not null and (m.retirement is null or not m.retirement) and r.round is not null and r.round > 0 and m.start_date < '{{END_DATE}}'::date and m.start_date >= '{{START_DATE}}'::date and not m.round like '%%Qualifying%%' 
    '''.replace('{{END_DATE}}', str(test_season)).replace('{{START_DATE}}', str(start_year))
    if not keep_nulls:
        sql_str = sql_str + '        and m.retirement is not null '
    sql = pd.read_sql(sql_str, conn)
    sql = sql[attributes].astype(np.float64, errors='ignore')
    print('Data shape:', sql.shape)
    return sql


def get_all_data(all_attributes, test_season='2017-01-01', num_test_years=1, start_year='2005-01-01', tournament=None):
    date = datetime.date(int(test_season[0:4])-num_test_years, int(test_season[5:7]), int(test_season[8:10])).strftime('%Y-%m-%d')
    data = load_data(all_attributes, test_season=date, start_year=start_year, keep_nulls=False)
    test_data = load_data(all_attributes, test_season=test_season, start_year=date, keep_nulls=tournament is not None)
    if tournament is not None:
        #data = data[data.tournament==tournament]
        test_data = test_data[test_data.tournament==tournament]
        print('data size after tournament filter:', data.shape, test_data.shape)
    return data, test_data


# previous year quality
input_attributes0 = [
    'h2h_prior_win_percent',
    'tourney_hist_avg_round',
    'prev_2year_avg_round',
    'prior_2year_match_recovery',
    'prev_2year_win_percent',
    'prev_year_victory_closeness',
    'prev_year_loss_closeness',

    # prior quarter
    #'prior_quarter_encounters',
    'prior_quarter_victories',
    'prior_quarter_losses',

    # player qualities
    'elo_score_weighted',
    'surface_experience',
    'injuries',
    'first_match_date',
    'best_year',

    # match stats
    'major_encounters',
    'master_encounters',
    'encounters_250',
    'challenger_encounters',
    'tiebreak_win_percent',
    'mean_faults',
    'mean_aces',
    'mean_service_points_won',
    'mean_return_points_made',

    # previous match
    'previous_games_total2',
    'had_qualifier',
    'wild_card',
    'seeded',
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

meta_attributes = ['challenger', 'start_date', 'first_round', 'tournament_rank', 'clay', 'hard', 'grass', 'prev_year_prior_encounters', 'opp_prev_year_prior_encounters', 'player_id', 'opponent_id', 'tournament', 'year', 'grand_slam', 'round_num', 'court_surface']
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


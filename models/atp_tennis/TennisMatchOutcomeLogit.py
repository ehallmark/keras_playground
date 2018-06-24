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
            m.games_won-m.games_against as spread,
            case when m.player_victory then 1.0 else 0.0 end as y, 
            case when m.court_surface = 'Clay' then 1.0 else 0.0 end as clay,
            case when m.court_surface = 'Grass' then 1.0 else 0.0 end as grass,
            m.year as year,
            m.player_id as player_id,
            m.opponent_id as opponent_id,
            m.tournament as tournament,
            case when m.tournament in ('roland-garros','wimbledon','us-open','australian-open')
                then 1.0 else 0.0 end as grand_slam,
            coalesce(h2h.prior_encounters,0) as h2h_prior_encounters,
            coalesce(h2h.prior_victories,0) as h2h_prior_victories,
            coalesce(h2h.prior_losses,0) as h2h_prior_losses,
            coalesce(h2h.prior_victories,0)-coalesce(h2h.prior_losses,0) as h2h_prior_win_percent,        
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
            coalesce(mean_opp.duration,90.0) as mean_opp_duration,
            coalesce(var.duration,1000.0) as var_duration,
            coalesce(var_opp.duration,1000.0) as var_opp_duration,
            coalesce(h2h2.both_win, 0) as prev_h2h2_both_win,
            coalesce(h2h2.both_lost, 0) as prev_h2h2_both_lost,
            coalesce(h2h2.wins_player, 0) as prev_h2h2_wins_player,
            coalesce(h2h2.wins_opponent,0) as prev_h2h2_wins_opponent,
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
            case when pc.date_of_birth is null then (select avg_age from avg_player_characteristics)
                else m.year - extract(year from pc.date_of_birth) end as age,
            case when pc_opp.date_of_birth is null then (select avg_age from avg_player_characteristics)
                else m.year - extract(year from pc_opp.date_of_birth) end as opp_age,
            case when pc.turned_pro is null then (select avg_experience from avg_player_characteristics)
                else m.year - pc.turned_pro end as experience,
            case when pc_opp.turned_pro is null then (select avg_experience from avg_player_characteristics)
                else m.year - pc_opp.turned_pro end as opp_experience,
            case when m.round='Round of 64' then 1
                when m.round='Round of 32' then 2
                when m.round='Round of 16' then 3
                when m.round='Quarter-Finals' then 4
                when m.round='Semi-Finals' then 5
                when m.round='Finals' then 6
                else 0
            end as round,
        coalesce(elo.score1,0) as elo_score,
        coalesce(elo.score2,0) as opp_elo_score
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
        left outer join atp_matches_prior_match as prior_match
            on ((m.player_id,m.opponent_id,m.tournament,m.year)=(prior_match.player_id,prior_match.opponent_id,prior_match.tournament,prior_match.year))
        left outer join atp_matches_prior_match as prior_match_opp
            on ((m.opponent_id,m.player_id,m.tournament,m.year)=(prior_match_opp.player_id,prior_match_opp.opponent_id,prior_match_opp.tournament,prior_match_opp.year))
        left outer join atp_matches_prior_h2h_level2 as h2h2
            on ((m.player_id,m.opponent_id,m.tournament,m.year)=(h2h2.player_id,h2h2.opponent_id,h2h2.tournament,h2h2.year))
        left outer join atp_player_opponent_score as elo
            on ((m.player_id,m.opponent_id,m.tournament,m.year)=(elo.player_id,elo.opponent_id,elo.tournament,elo.year))
        where m.year <= {{END_DATE}} and m.year >= {{START_DATE}} 
    '''.replace('{{END_DATE}}', str(test_season)).replace('{{START_DATE}}', str(start_year))
    if not keep_nulls:
        sql_str = sql_str + '        and m.first_serve_attempted > 0'
    sql = pd.read_sql(sql_str, conn)
    sql = sql[attributes].astype(np.float64, errors='ignore')
    test_data = sql[sql.year == test_season]
    sql = sql[sql.year != test_season]
    return sql, test_data


def get_all_data(all_attributes, test_season=2017, start_year=2003, tournament=None, include_spread=False):
    all_data = load_data(all_attributes, test_season=test_season, start_year=start_year, keep_nulls=tournament is not None)
    data, test_data = all_data
    if tournament is not None:
        print('Tournament: ', tournament, ' ...using test data only')
        data = data[(data.tournament==tournament)]
        test_data = test_data[(test_data.tournament==tournament)]
        print('data size:', data.shape, test_data.shape)
    return data, test_data


input_attributes = [
        'prev_h2h2_wins_player',
        'prev_h2h2_wins_opponent',
        #'mean_duration',
        #'mean_opp_duration',
        'mean_return_points_made',
        'mean_opp_return_points_made',
        'mean_second_serve_points_made',
        'mean_opp_second_serve_points_made',
        'h2h_prior_win_percent',
        'prev_year_prior_encounters',
        'opp_prev_year_prior_encounters',
        #'prev_year_avg_round',
        #'opp_prev_year_avg_round',
        #'opp_tourney_hist_avg_round',
        #'tourney_hist_avg_round',
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
        #'duration_prev_match',
        #'opp_duration_prev_match',
        'elo_score',
        'opp_elo_score'
    ]


input_attributes_spread = [
        'prev_h2h2_wins_player',
        'prev_h2h2_wins_opponent',
        'mean_return_points_made',
        'mean_opp_return_points_made',
        'prev_year_prior_encounters',
        'opp_prev_year_prior_encounters',
        'tourney_hist_prior_encounters',
        'opp_tourney_hist_prior_encounters',
        'mean_break_points_made',
        'mean_opp_break_points_made',
        'tiebreak_win_percent',
        'opp_tiebreak_win_percent',
        'surface_experience',
        'opp_surface_experience',
        'experience',
        'opp_experience',
        'age',
        'opp_age',
        'height',
        'opp_height',
        'elo_score',
        'opp_elo_score'
    ]

y = 'y'
y_spread = 'spread'
all_attributes = list(input_attributes)
all_attributes.append('grand_slam')
all_attributes.append('round')
all_attributes.append(y)
all_attributes.append(y_spread)
meta_attributes = ['player_id', 'opponent_id', 'tournament', 'year']
for attr in input_attributes_spread:
    if attr not in all_attributes:
        all_attributes.append(attr)
for meta in meta_attributes:
    if meta not in all_attributes:
        all_attributes.append(meta)

if __name__ == '__main__':
    train_spread_model = False
    train_outcome_model = True
    sql, test_data = load_data(all_attributes, test_season=2011, start_year=1996)
    if train_outcome_model:
        model_file = 'tennis_match_outcome_logit.statmodel'
        # print('Attrs: ', sql[all_attributes][0:20])
        # model to predict the total score (h_pts + a_pts)
        results = smf.logit(y+' ~ '+'+'.join(input_attributes), data=sql).fit()
        print(results.summary())
        binary_correct, n, binary_percent, avg_error = test_model(results, test_data, test_data[y])

        print('Correctly predicted: '+str(binary_correct)+' out of '+str(n) +
              ' ('+to_percentage(binary_percent)+')')
        print('Average error: ', to_percentage(avg_error))
        results.save(model_file)

    if train_spread_model:
        model_file = 'tennis_match_spread_logit.statmodel'
        # print('Attrs: ', sql[all_attributes][0:20])
        # model to predict the total score (h_pts + a_pts)
        results = smf.ols(y_spread + ' ~ ' + '+'.join(input_attributes_spread), data=sql).fit()
        print(results.summary())
        n, avg_error = test_model(results, test_data, test_data[y], include_binary=False)
        print('Average error: ', avg_error)
        results.save(model_file)

    exit(0)


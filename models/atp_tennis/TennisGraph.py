from pgmpy.models import BayesianModel, FactorGraph
import pandas as pd
from sqlalchemy import create_engine
from pgmpy.estimators import ParameterEstimator, BayesianEstimator
from pgmpy.factors.continuous import LinearGaussianCPD, ContinuousFactor

conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
sql_str = '''
    select h2h.prior_victories::float/greatest(h2h.prior_encounters,1) as h2h,
        prior1.prior_victories::float/greatest(prior1.prior_encounters,1) as prev_win1,
        prior2.prior_victories::float/greatest(prior2.prior_encounters,1) as prev_win2,
        m.player_victory as outcome
    from atp_matches_individual as m 
    left outer join atp_matches_prior_h2h as h2h
    on (m.player_id,m.opponent_id,m.tournament,m.year)=(h2h.player_id,h2h.opponent_id,h2h.tournament,h2h.year)
    left outer join atp_matches_prior_year as prior1
    on (m.player_id,m.tournament,m.year)=(prior1.player_id,prior1.tournament,prior1.year)
    left outer join atp_matches_prior_year as prior2
    on (m.opponent_id,m.tournament,m.year)=(prior2.player_id,prior2.tournament,prior2.year)
    where m.year >= 2003 and m.year <=2017
'''
sql = pd.read_sql(sql_str, conn)

edges = [
    ('prev_win1', 'outcome'),
    ('prev_win2', 'outcome'),
    ('h2h', 'outcome'),
]

model = FactorGraph(edges)

factors = [
    ContinuousFactor(('h2h',), lambda x: ),

}


model.add_factors(factors)

pe = BayesianEstimator(model, sql)
print("\n", pe.state_counts('outcome'))  # unconditional
print("\n", pe.state_counts('h2h'))  # conditional on fruit and size


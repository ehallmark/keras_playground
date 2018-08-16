import psycopg2
from threading import Thread
from sqlalchemy import create_engine
import pandas as pd


def table_name(i):
    return 'atp_matches_quarter'+str(i)


def attributes_for(i, include_opp=True, opp_only=False):
    attrs = [
        'q' + str(i) + '.' + 'victories as victories_q' + str(i),
        'q' + str(i) + '.' + 'losses as losses_q' + str(i),
        'q' + str(i) + '.' + 'encounters as encounters_q' + str(i),
        'q' + str(i) + '.' + 'match_closeness as match_closeness_q' + str(i),
        'q' + str(i) + '.' + 'games_per_set as games_per_set_q' + str(i),
        'q' + str(i) + '.' + 'num_tournaments as num_tournaments_q' + str(i),
        'q' + str(i) + '.' + 'avg_tournament_rank as avg_tournament_rank_q' + str(i),
        'q' + str(i) + '.' + 'atp_points as atp_points_q' + str(i),
        'q' + str(i) + '.' + 'better_encounters as better_encounters_q' + str(i),
        'q' + str(i) + '.' + 'worse_encounters as worse_encounters_q' + str(i),
        'q' + str(i) + '.' + 'spread_avg as spread_avg_q' + str(i),
        'q' + str(i) + '.' + 'spread_var as spread_var_q' + str(i),
        'q' + str(i) + '.' + 'grass_percent as grass_percent_q'+str(i),
        'q' + str(i) + '.' + 'clay_percent as clay_percent_q' + str(i),
        'q' + str(i) + '.' + 'local_percent as local_percent_q' + str(i),
    ]
    if include_opp or opp_only:
        opp_attrs = ['opp_'+attr.replace(' as ', ' as opp_') for attr in attrs]
        if opp_only:
            attrs = opp_attrs
        else:
            attrs = attrs + opp_attrs
    return attrs


def attribute_names_for(i, include_opp=True, opp_only=False):
    attrs = [
        'victories_q' + str(i),
        'losses_q' + str(i),
        'encounters_q' + str(i),
        'match_closeness_q' + str(i),
        'games_per_set_q' + str(i),
        'num_tournaments_q' + str(i),
        'avg_tournament_rank_q' + str(i),
        'atp_points_q' + str(i),
        'better_encounters_q' + str(i),
        'worse_encounters_q' + str(i),
        'spread_avg_q' + str(i),
        'spread_var_q' + str(i),
        'grass_percent_q' + str(i),
        'clay_percent_q' + str(i),
        'local_percent_q' + str(i),
    ]
    if include_opp or opp_only:
        opp_attrs = ['opp_'+attr for attr in attrs]
        if opp_only:
            attrs = opp_attrs
        else:
            attrs = attrs + opp_attrs
    return attrs


def attribute_definitions_for(i, include_opp=True, opp_only=False):
    attrs = [
        'victories_q' + str(i) + ' float',
        'losses_q' + str(i) + ' float',
        'encounters_q' + str(i) + ' float',
        'match_closeness_q' + str(i) + ' float',
        'games_per_set_q' + str(i) + ' float',
        'num_tournaments_q' + str(i) + ' float',
        'avg_tournament_rank_q' + str(i) + ' float',
        'atp_points_q' + str(i) + ' float',
        'better_encounters_q' + str(i) + ' float',
        'worse_encounters_q' + str(i) + ' float',
        'spread_avg_q' + str(i) + ' float',
        'spread_var_q' + str(i) + ' float',
        'grass_percent_q' + str(i) + ' float',
        'clay_percent_q' + str(i) + ' float',
        'local_percent_q' + str(i) + ' float',
    ]
    if include_opp or opp_only:
        opp_attrs = ['opp_'+attr for attr in attrs]
        if opp_only:
            attrs = opp_attrs
        else:
            attrs = attrs + opp_attrs
    return attrs


def join_str(i):
    return ' left outer join '+table_name(i)+' as q'+str(i) + \
           '  on ((m.player_id,m.tournament,m.start_date)=(q'+str(i)+'.player_id,q'+str(i)+'.tournament,q'+str(i)+'.start_date))' + \
           ' left outer join '+table_name(i)+' as opp_q'+str(i) + \
           '  on ((m.opponent_id,m.tournament,m.start_date)=(opp_q'+str(i)+'.player_id,opp_q'+str(i)+'.tournament,opp_q'+str(i)+'.start_date))'


def run(i):
    conn = psycopg2.connect("postgresql://localhost/ib_db?user=postgres&password=password")
    cursor = conn.cursor()
    sql = '''
        drop table if exists atp_matches_quarter{{N}};
    '''.replace("{{N}}", str(i))
    cursor.execute(sql)
    print("Dropped table...", i)
    sql = '''
        create table atp_matches_quarter{{N}} (
            player_id text not null,
            tournament text not null,
            start_date date not null,
            victories float not null,
            losses float not null,
            encounters float not null,
            match_closeness float not null,
            games_per_set float not null,
            num_tournaments float not null,
            avg_tournament_rank float not null,
            atp_points float not null,
            better_encounters float not null,
            worse_encounters float not null,
            spread_avg float not null,
            spread_var float not null,
            clay_percent float not null,
            grass_percent float not null,
            local_percent float not null,
            primary key(player_id, tournament, start_date)
        );
    '''.replace("{{N}}", str(i))
    cursor.execute(sql)
    print("Created table...", i)
    sql = '''
        insert into atp_matches_quarter{{N}} (
            select distinct on (player_id,start_date,tournament) * from (
                select m.player_id, m.tournament, m.start_date,
                sum(case when m2.player_victory then 1 else 0 end)/9.0 as victories,
                sum(case when m2.player_victory then 0 else 1 end)/9.0 as losses,
                count(m2.*)::double precision/18.0 as encounters,
                coalesce(sum(m2.sets_won::double precision/greatest(1,coalesce(m2.num_sets,0))), 0) as match_closeness,
                sum(m2.games_won+m2.games_against)/greatest(12.0*sum(m2.num_sets),1) as games_per_set,
                count(distinct m2.tournament)/9.0 as num_tournaments,
                sum(coalesce(t2.masters,0))/(greatest(1,1000.0*count(m2.*))) as avg_tournament_rank,
                coalesce(sum(case when m2.player_victory then t2.masters else 0 end), 0)/3000.0 as atp_points,
                sum(case when t.masters <= t2.masters then 1.0 else 0.0 end)/18.0 as better_encounters,
                sum(case when t.masters >= t2.masters then 1.0 else 0.0 end)/18.0 as worse_encounters,
                sum(m2.games_won-m2.games_against)::double precision/(12.0*greatest(1,count(m2.*))) as spread_avg,
                coalesce(var_samp(m2.games_won-m2.games_against), 0)/18.0 as spread_var,
                coalesce(sum(case when t2.court_surface='Grass' then 1.0 else 0.0 end)/count(m2.*), 0) as grass_percent,
                coalesce(sum(case when t2.court_surface='Clay' then 1.0 else 0.0 end)/count(m2.*), 0) as clay_percent,
                coalesce(sum(case when player1.country is null then 0.0 else case when player1.country = coalesce(country.code, t2.location) then 1.0 else 0.0 end end)/count(m2.*), 0.0) as local_percent
                from atp_matches_individual as m
                join atp_matches_individual as m2
                on (
                    m.player_id=m2.player_id
                    and m.start_date - interval '{{MONTHS_START}} months' > m2.start_date
                    and m.start_date - interval '{{MONTHS_END}} months' < m2.start_date
                )
                join atp_tournament_dates as t
                on ((t.start_date,t.tournament)=(m.start_date,m.tournament))
                join atp_tournament_dates as t2
                on ((t2.start_date,t2.tournament)=(m2.start_date,m2.tournament))
                join atp_player_characteristics as player1
                on ((m2.player_id,m2.tournament,m2.start_date)=(player1.player_id,player1.tournament,player1.start_date))
                left outer join atp_countries as country 
                on (t2.location = country.name)
                where m2.player_victory is not null and t.masters > 0
                group by m.player_id,m.opponent_id,m.start_date,m.tournament
            ) as temp order by player_id,start_date,tournament,random()            
        );
    '''.replace("{{N}}", str(i)).replace("{{MONTHS_START}}", str(i * 3)).replace("{{MONTHS_END}}", str(i * 3 + 3))
    print('Sql:', sql)
    cursor.execute(sql)
    conn.commit()
    print("Executed query...", i)
    cursor.close()


num_prior_quarters = 16

all_attributes = []
for i in range(num_prior_quarters):
    for attr in attributes_for(i, include_opp=True):
        all_attributes.append(attr)

all_attribute_names = []
for i in range(num_prior_quarters):
    for attr in attribute_names_for(i, include_opp=True):
        all_attribute_names.append(attr)


def load_data(date='2000-01-01', end_date=None, include_null=False):
    conn = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
    sql = 'select case when m.player_victory then 1.0 else 0.0 end as y2, m.start_date as start_date,m.tournament as tournament,m.player_id as player_id,m.opponent_id as opponent_id,' + ','.join(all_attribute_names) + ' from atp_matches_quarters_all as m '  \
          + ' where m.start_date >= \'{{DATE}}\'::date'.replace('{{DATE}}', date)
    if not include_null:
        sql += ' and m.player_victory is not null'
    if end_date is not None:
        sql += ' and m.start_date < \''+str(end_date) + '\'::date '
    data = pd.read_sql(sql, conn)
    data.fillna(value=0., inplace=True)
    return data


if __name__ == '__main__':
    print('Running aggregations in parallel')
    threads = []
    for i in range(num_prior_quarters):
        thread = Thread(target=run, args=(i,))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    # build join table
    print('Building join table...')
    conn = psycopg2.connect("postgresql://localhost/ib_db?user=postgres&password=password")
    cursor = conn.cursor()
    sql = '''
        drop table if exists atp_matches_quarters_all;
    '''
    cursor.execute(sql)
    print("Dropped table...")
    grouped_attr_names = []
    for i in range(num_prior_quarters):
        for attr in attribute_definitions_for(i, include_opp=True):
            grouped_attr_names.append(attr)

    sql = '''
        create table atp_matches_quarters_all (
            player_victory boolean,
            player_id text not null,
            opponent_id text not null,
            tournament text not null,
            start_date date not null,
    ''' + ', \n'.join(grouped_attr_names) + \
    ''',
            primary key(player_id, opponent_id, tournament, start_date)
        );
    '''
    cursor.execute(sql)
    sql = 'insert into atp_matches_quarters_all (' + \
          'select m.player_victory, m.player_id, m.opponent_id, m.tournament, m.start_date,' + ','.join(all_attributes) + ' from atp_matches_individual as m ' + ' '.join([join_str(i) for i in range(num_prior_quarters)]) + \
        ' )'
    cursor.execute(sql)
    conn.commit()
    cursor.close()


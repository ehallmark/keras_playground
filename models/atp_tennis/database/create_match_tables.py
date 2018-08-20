import psycopg2
from threading import Thread
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
file_prefix = '/home/ehallmark/repos/keras_playground/models/atp_tennis/database/'


class TableCreator:

    def __init__(self, prefix, table, num_tables, join_table_name, time_period, where_str):
        self.prefix = prefix
        self.time_period = time_period
        self.where_str = where_str
        self.num_tables = num_tables
        self.join_table_name = join_table_name
        self.table = table

    def table_name(self, i):
        return self.table + str(i)

    def attributes_for(self, i, include_opp=True, opp_only=False):
        attrs = [
            self.prefix + str(i) + '.' + 'victories as victories_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'losses as losses_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'encounters as encounters_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'match_closeness as match_closeness_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'games_per_set as games_per_set_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'num_tournaments as num_tournaments_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'avg_tournament_rank as avg_tournament_rank_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'atp_points as atp_points_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'better_encounters as better_encounters_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'worse_encounters as worse_encounters_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'spread_avg as spread_avg_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'spread_var as spread_var_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'grass_percent as grass_percent_' + self.prefix+str(i),
            self.prefix + str(i) + '.' + 'clay_percent as clay_percent_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'local_percent as local_percent_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'qualifier_percent as qualifier_percent_' + self.prefix + str(i),
        ]
        if include_opp or opp_only:
            opp_attrs = ['opp_'+attr.replace(' as ', ' as opp_') for attr in attrs]
            if opp_only:
                attrs = opp_attrs
            else:
                attrs = attrs + opp_attrs
        return attrs

    def attribute_names_for(self, i, include_opp=True, opp_only=False):
        attrs = [
            'victories_' + self.prefix + str(i),
            'losses_' + self.prefix + str(i),
            'encounters_' + self.prefix + str(i),
            'match_closeness_' + self.prefix + str(i),
            'games_per_set_' + self.prefix + str(i),
            'num_tournaments_' + self.prefix + str(i),
            'avg_tournament_rank_' + self.prefix + str(i),
            'atp_points_' + self.prefix + str(i),
            'better_encounters_' + self.prefix + str(i),
            'worse_encounters_' + self.prefix + str(i),
            'spread_avg_' + self.prefix + str(i),
            'spread_var_' + self.prefix + str(i),
            'grass_percent_' + self.prefix + str(i),
            'clay_percent_' + self.prefix + str(i),
            'local_percent_' + self.prefix + str(i),
            'qualifier_percent', + self.prefix + str(i),
        ]
        if include_opp or opp_only:
            opp_attrs = ['opp_'+attr for attr in attrs]
            if opp_only:
                attrs = opp_attrs
            else:
                attrs = attrs + opp_attrs
        return attrs

    def attribute_definitions_for(self, i, include_opp=True, opp_only=False):
        attrs = [attr + ' float' for attr in self.attribute_names_for(i, include_opp=include_opp, opp_only=opp_only)]
        return attrs

    def join_str(self, i):
        return ' left outer join '+self.table_name(i)+' as '+self.prefix+str(i) + \
               '  on ((m.player_id,m.tournament,m.start_date)=('+self.prefix+str(i)+'.player_id,'+self.prefix+str(i)+'.tournament,'+self.prefix+str(i)+'.start_date))' + \
               ' left outer join '+self.table_name(i)+' as opp_'+self.prefix+str(i) + \
               '  on ((m.opponent_id,m.tournament,m.start_date)=(opp_'+self.prefix+str(i)+'.player_id,opp_'+self.prefix+str(i)+'.tournament,opp_'+self.prefix+str(i)+'.start_date))'

    def run(self, i):
        time_period = self.time_period
        conn = psycopg2.connect("postgresql://localhost/ib_db?user=postgres&password=password")
        cursor = conn.cursor()
        sql = '''
            drop table if exists {{N}};
        '''.replace("{{N}}", self.table+str(i))
        cursor.execute(sql)
        print("Dropped table...", i)
        sql = '''
            create table {{N}} (
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
                qualifier_percent float not null,
                primary key(player_id, tournament, start_date)
            );
        '''.replace("{{N}}", self.table+str(i))
        cursor.execute(sql)
        print("Created table...", i)
        sql = '''
            insert into {{N}} (
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
                    coalesce(sum(case when player1.country is null then 0.0 else case when player1.country = coalesce(country.code, t2.location) then 1.0 else 0.0 end end)/count(m2.*), 0.0) as local_percent,
                    coalesce(sum(case when m2.round like '%%Qualifying%%' then 1.0 else 0.0 end)/greatest(1, count(m2.*)), 0.0)
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
                    where m2.player_victory is not null and t.masters > 0 and ({{WHERE_STR}})
                    group by m.player_id,m.opponent_id,m.start_date,m.tournament
                ) as temp order by player_id,start_date,tournament,random()            
            );
        '''.replace("{{N}}", self.table+str(i)).replace('{{WHERE_STR}}', self.where_str).replace("{{MONTHS_START}}", str(i * time_period)).replace("{{MONTHS_END}}", str(i * time_period + time_period))
        print('Sql:', sql)
        cursor.execute(sql)
        conn.commit()
        print("Executed query...", i)
        cursor.close()

    def build_tables(self):
        print('Running aggregations in parallel')
        threads = []
        for i in range(self.num_tables):
            thread = Thread(target=self.run, args=(i,))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        # build join table
        print('Building join table...')
        sql = 'select m.player_victory, m.player_id, m.opponent_id, m.tournament, m.start_date,' + ','.join(
            self.all_attributes()) + ' from atp_matches_individual as m ' + ' '.join(
            [self.join_str(i) for i in range(self.num_tables)])
        df = pd.read_sql(sql, engine)
        df.to_hdf(file_prefix+self.join_table_name+'.hdf', self.join_table_name, mode='w')
        print("Data size:", df.shape[0])

    def all_attributes(self):
        all_attributes = []
        for i in range(self.num_tables):
            for attr in self.attributes_for(i, include_opp=True):
                all_attributes.append(attr)

        return all_attributes

    def all_attribute_names(self):
        all_attribute_names = []
        for i in range(self.num_tables):
            for attr in self.attribute_names_for(i, include_opp=True):
                all_attribute_names.append(attr)
        return all_attribute_names

    def load_data(self, date=None, end_date=None, include_null=False):
        print('Loading table:', self.join_table_name)
        df = pd.read_hdf(file_prefix+self.join_table_name+'.hdf', self.join_table_name)
        #if not include_null:
        #    df = df[np.df.player_victory is not None]
        if end_date is not None:
            df = df[df.start_date < end_date]
        if date is not None:
            df = df[df.start_date >= date]
        print('Loaded.')
        return df


quarter_tables = TableCreator(prefix='q', table='atp_matches_quarter', num_tables=16,
                              join_table_name='atp_matches_quarters_all', time_period=3, where_str="'t'")

junior_tables = TableCreator(prefix='jun', table='atp_matches_juniors', num_tables=1,
                              join_table_name='atp_matches_juniors_all', time_period=60, where_str="t2.masters = 0")

itf_tables = TableCreator(prefix='itf', table='atp_matches_itf', num_tables=1,
                              join_table_name='atp_matches_itf_all', time_period=60, where_str="t2.masters = 25")

challenger_tables = TableCreator(prefix='ch', table='atp_matches_challenger', num_tables=1,
                              join_table_name='atp_matches_challenger_all', time_period=60, where_str="t2.masters = 100")

pro_tables = TableCreator(prefix='pro', table='atp_matches_pro', num_tables=1,
                              join_table_name='atp_matches_pro_all', time_period=60, where_str="t2.masters > 200")

if __name__ == '__main__':
    threads = []
    for func in [junior_tables.build_tables, itf_tables.build_tables,
                 challenger_tables.build_tables, pro_tables.build_tables]:
        thread = Thread(target=func, args=())
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print('Starting quarterly tables...')
    quarter_tables.build_tables()

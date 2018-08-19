import psycopg2
from threading import Thread
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from models.atp_tennis.database.create_match_tables import TableCreator

engine = create_engine("postgresql://localhost/ib_db?user=postgres&password=password")
file_prefix = '/home/ehallmark/repos/keras_playground/models/atp_tennis/database/'


class DailyTable(TableCreator):

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
        ]
        if include_opp or opp_only:
            opp_attrs = ['opp_'+attr for attr in attrs]
            if opp_only:
                attrs = opp_attrs
            else:
                attrs = attrs + opp_attrs
        return attrs

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
                primary key(player_id, opponent_id, tournament, start_date)
            );
        '''.replace("{{N}}", self.table+str(i))
        cursor.execute(sql)
        print("Created table...", i)
        sql = '''
            insert into {{N}} (
                select distinct on (player_id,opponent_id,start_date,tournament) * from (
                    select
                        p1.player_id,
                        p1.tournament,
                        p1.start_date,
                        p1.opponent_id,
                        p1.opponent_name,
                        p2.opponent_id as prev_opponent_id,
                        p2.opponent_name as prev_opponent_name,
                        round_num,
                        p2.num_sets,
                        p2.sets_won,
                        p2.games_won,
                        p2.games_against,
                        p2.tiebreaks_won,
                        p2.tiebreaks_total,
                        p2.duration,
                        p2.player_victory
                    from atp_matches_individual as p1
                    join atp_matches_round as r1
                    on ((p1.start_date,p1.tournament,p1.player_id,p1.opponent_id)=(r1.start_date,r1.tournament,r1.player_id,r1.opponent_id))
                    join (
                        select p2.*,r2.round as round_num from atp_matches_individual as p2
                        join atp_matches_round as r2
                        on ((p2.start_date,p2.tournament,p2.player_id,p2.opponent_id)=(r2.start_date,r2.tournament,r2.player_id,r2.opponent_id))
                    ) as p2
                    on (
                        (p1.player_id,p1.tournament,p1.start_date)=(p2.player_id,p2.tournament,p2.start_date) and
                        r1.round = round_num + 1
                    )
                    where p2.player_victory is not null 
                ) as temp order by player_id,temp.opponent_id,start_date,tournament,random()
                );
        '''.replace("{{N}}", self.table+str(i)).replace('{{WHERE_STR}}', self.where_str).replace("{{START}}", str(i * time_period)).replace("{{END}}", str(i * time_period + time_period))
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
        #conn = psycopg2.connect("postgresql://localhost/ib_db?user=postgres&password=password")
        #cursor = conn.cursor()
        #sql = '''
        #      drop table if exists {{TABLE}};
        #  '''.replace('{{TABLE}}', self.join_table_name)
        #cursor.execute(sql)
        #print("Dropped table...")
        #grouped_attr_names = []
        #for i in range(self.num_tables):
        #    for attr in quarter_tables.attribute_definitions_for(i, include_opp=True):
        #        grouped_attr_names.append(attr)
        #sql = '''
        #     create table {{TABLE}} (
        #          player_victory boolean,
        #          player_id text not null,
        #          opponent_id text not null,
        #          tournament text not null,
        #          start_date date not null,
        #  '''.replace('{{TABLE}}', self.join_table_name) + ', \n'.join(grouped_attr_names) + \
        #      ''',
        #              primary key(player_id, opponent_id, tournament, start_date)
        #          );
        #      '''
        #cursor.execute(sql)
        sql = 'select m.player_victory, m.player_id, m.opponent_id, m.tournament, m.start_date,' + ','.join(
            self.all_attributes()) + ' from atp_matches_individual as m ' + ' '.join(
            [self.join_str(i) for i in range(self.num_tables)])
        df = pd.read_sql(sql, engine)
        df.to_hdf(file_prefix+self.join_table_name+'.hdf', self.join_table_name, mode='w')
        print("Data size:", df.shape[0])
        #conn.commit()
        #cursor.close()

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
    print('Starting quarterly tables...')
    quarter_tables.build_tables()
    print("Starting junior tables...")
    junior_tables.build_tables()
    print("Starting itf tables...")
    itf_tables.build_tables()
    print("Starting challenger tables...")
    challenger_tables.build_tables()
    print("Starting pro tables...")
    pro_tables.build_tables()


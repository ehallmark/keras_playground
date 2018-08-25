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
            self.prefix + str(i) + '.' + 'victory as victory_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'sets_won as sets_won_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'tournament_rank as tournament_rank_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'round_num as round_num_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'games_won as games_won_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'elo1 as elo1_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'clay as clay_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'grass as grass_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'hard as hard_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'local as local_' + self.prefix + str(i),
            self.prefix + str(i) + '.' + 'qualifier as qualifier_' + self.prefix + str(i),
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
            'victory_' + self.prefix + str(i),
            'sets_won_' + self.prefix + str(i),
            'tournament_rank_' + self.prefix + str(i),
            'round_num_' + self.prefix + str(i),
            'games_won_' + self.prefix + str(i),
            'elo1_' + self.prefix + str(i),
            'clay_' + self.prefix + str(i),
            'grass_' + self.prefix + str(i),
            'hard_' + self.prefix + str(i),
            'local_' + self.prefix + str(i),
            'qualifier_' + self.prefix + str(i),
        ]
        if include_opp or opp_only:
            opp_attrs = ['opp_'+attr for attr in attrs]
            if opp_only:
                attrs = opp_attrs
            else:
                attrs = attrs + opp_attrs
        return attrs

    def join_str(self, i):
        return ' left outer join '+self.table_name(i)+' as '+self.prefix+str(i) + \
               '  on ((m.player_id,m.opponent_id,m.tournament,m.start_date)=('+self.prefix+str(i)+'.player_id,'+self.prefix+str(i)+'.opponent_id,'+self.prefix+str(i)+'.tournament,'+self.prefix+str(i)+'.start_date))' + \
               ' left outer join '+self.table_name(i)+' as opp_'+self.prefix+str(i) + \
               '  on ((m.opponent_id,m.player_id,m.tournament,m.start_date)=(opp_'+self.prefix+str(i)+'.player_id,opp_'+self.prefix+str(i)+'.opponent_id,opp_'+self.prefix+str(i)+'.tournament,opp_'+self.prefix+str(i)+'.start_date))'

    def run(self, i):
        conn = psycopg2.connect("postgresql://localhost/ib_db?user=postgres&password=password")
        cursor = conn.cursor()
        sql = '''
            drop table if exists {{N}};
        '''.replace("{{N}}", self.table+str(i))
        cursor.execute(sql)
        print("Dropped daily table...", i)
        sql = '''
            create table {{N}} (
                player_id text not null,
                opponent_id text not null,
                tournament text not null,
                start_date date not null,
                victory float not null,
                sets_won float not null,
                tournament_rank float not null,
                round_num float not null,
                games_won float not null,
                elo1 float not null,
                clay float not null,
                grass float not null,
                hard float not null,
                local float not null,
                qualifier float not null,
                primary key(player_id, opponent_id, tournament, start_date)
            );
        '''.replace("{{N}}", self.table+str(i))
        cursor.execute(sql)
        print("Created table...", i)
        sql = '''
            insert into {{N}} (
            select distinct on (player_id, opponent_id, tournament, start_date) * from (
                select
                    p1.player_id,
                    p1.opponent_id,
                    p1.tournament,
                    p1.start_date,
                    case when p2.player_victory then 1.0 else 0.0 end,
                    p2.sets_won::float/greatest(1,p2.num_sets),
                    p2.masters / 2000.0,
                    p2.round_num::float/7.0,
                    p2.games_won::float/18.0,
                    coalesce(weighted_score1, 100.0)/1000.0,
                    case when p2.court_surface='Clay' then 1.0 else 0.0 end,
                    case when p2.court_surface='Grass' then 1.0 else 0.0 end,
                    case when p2.court_surface='Hard' then 1.0 else 0.0 end,
                    case when player1.country is null then 0.0 else case when player1.country = coalesce(country.code, p2.nation) then 1.0 else 0.0 end end as local,
                    case when p2.round like '%%Qualifying%%' then 1.0 else 0.0 end
                from atp_matches_match_history_linear as p1
                join  atp_matches_individual_with_meta as p2
                on ((
                        p1.player_id,
                        p1.tournament{{IDX}},                    
                        p1.start_date{{IDX}},
                        p1.round{{IDX}}
                    ) =(p2.player_id,p2.tournament,p2.start_date,p2.round_num))
                join atp_tournament_dates as t
                on ((t.start_date,t.tournament)=(p1.start_date,p1.tournament))
                join atp_player_characteristics as player1
                on ((p2.player_id,p2.tournament,p2.start_date)=(player1.player_id,player1.tournament,player1.start_date))
                left outer join atp_countries as country
                on (p2.nation = country.name)
                left outer join atp_player_opponent_score as elo
                on ((elo.player_id,elo.opponent_id,elo.tournament,elo.start_date)=(p2.player_id,p2.opponent_id,p2.tournament,p2.start_date))
                where p2.player_victory is not null and t.masters > 0 and {{WHERE_STR}}
            ) as temp order by player_id,opponent_id,tournament,start_date,random()
        )    
        '''.replace("{{N}}", self.table+str(i)).replace('{{WHERE_STR}}', self.where_str).replace("{{IDX}}", str(i + 1))
        print('Sql daily:', sql)
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
        print("Using sql: "+sql)
        df = pd.read_sql(sql, engine)
        df.to_hdf(file_prefix+self.join_table_name+'.hdf', self.join_table_name, mode='w')
        print("Data size:", df.shape[0])


daily_tables = DailyTable(prefix='dly', table='atp_matches_daily', num_tables=16,
                              join_table_name='atp_matches_daily_all', time_period=None, where_str="'t'")


if __name__ == '__main__':
    print('Starting tables...')
    daily_tables.build_tables()


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
            self.prefix + str(i) + '.' + 'grass_percent as grass_percent_' + self.prefix + str(i),
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
                select
                    p1.player_id, p1.tournament, p1.start_date,
                    sum(case when p2.player_victory then 1 else 0 end)/9.0 as victories,
                    sum(case when p2.player_victory then 0 else 1 end)/9.0 as losses,
                    count(p2.*)::double precision/18.0 as encounters,
                    coalesce(sum(p2.sets_won::double precision/greatest(1,coalesce(p2.num_sets,0))), 0) as match_closeness,
                    sum(p2.games_won+p2.games_against)/greatest(12.0*sum(p2.num_sets),1) as games_per_set,
                    count(distinct p2.tournament)/9.0 as num_tournaments,
                    sum(coalesce(t2.masters,0))/(greatest(1,1000.0*count(p2.*))) as avg_tournament_rank,
                    coalesce(sum(case when p2.player_victory then t2.masters else 0 end), 0)/3000.0 as atp_points,
                    sum(case when t.masters <= t2.masters then 1.0 else 0.0 end)/18.0 as better_encounters,
                    sum(case when t.masters >= t2.masters then 1.0 else 0.0 end)/18.0 as worse_encounters,
                    sum(p2.games_won-p2.games_against)::double precision/(12.0*greatest(1,count(p2.*))) as spread_avg,
                    coalesce(var_samp(p2.games_won-p2.games_against), 0)/18.0 as spread_var,
                    coalesce(sum(case when t2.court_surface='Grass' then 1.0 else 0.0 end)/count(p2.*), 0) as grass_percent,
                    coalesce(sum(case when t2.court_surface='Clay' then 1.0 else 0.0 end)/count(p2.*), 0) as clay_percent,
                    coalesce(sum(case when player1.country is null then 0.0 else case when player1.country = coalesce(country.code, t2.location) then 1.0 else 0.0 end end)/count(p2.*), 0.0) as local_percent,
                    coalesce(case when bool_or(p2.round like '%%Qualifying%%') then 1.0 else 0.0 end, 0.0)
                from atp_matches_tournaments_history as p1
                join atp_matches_individual as p2
                on (p1.previous[{{IDX}}] is not null and (
                        p1.player_id,
                        p1.previous[{{IDX}}].tournament,                    
                        p1.previous[{{IDX}}].start_date
                    )
                    =(p2.player_id,p2.tournament,p2.start_date))
                join atp_tournament_dates as t
                on ((t.start_date,t.tournament)=(p1.start_date,p1.tournament))
                join atp_tournament_dates as t2
                on ((t2.start_date,t2.tournament)=(p2.start_date,p2.tournament))
                join atp_player_characteristics as player1
                on ((p2.player_id,p2.tournament,p2.start_date)=(player1.player_id,player1.tournament,player1.start_date))
                left outer join atp_countries as country 
                on (t2.location = country.name)
                where p2.player_victory is not null and t.masters > 0 and {{WHERE_STR}}
                group by p1.player_id,p1.tournament,p1.start_date
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
        df = pd.read_sql(sql, engine)
        df.to_hdf(file_prefix+self.join_table_name+'.hdf', self.join_table_name, mode='w')
        print("Data size:", df.shape[0])


daily_tables = DailyTable(prefix='q', table='atp_matches_tourney_histories', num_tables=2,
                              join_table_name='atp_matches_tourney_histories_all', time_period=None, where_str="'t'")


if __name__ == '__main__':
    print('Starting tables...')
    daily_tables.build_tables()


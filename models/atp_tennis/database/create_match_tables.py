import psycopg2
from threading import Thread

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
            victories integer not null,
            losses integer not null,
            encounters integer not null,
            match_closeness double precision not null,
            games_per_set double precision,
            num_tournaments integer not null,
            avg_tournament_rank double precision not null,
            atp_points double precision not null,
            better_encounters integer not null,
            worse_encounters integer not null,
            primary key(player_id, tournament, start_date)
        );
    '''.replace("{{N}}", str(i))
    cursor.execute(sql)
    print("Created table...", i)
    sql = '''
        insert into atp_matches_quarter{{N}} (
            select distinct on (player_id,start_date,tournament) * from (
                select m.player_id, m.tournament, m.start_date,
                sum(case when m2.player_victory then 1 else 0 end) as victories,
                sum(case when m2.player_victory then 0 else 1 end) as losses,
                count(m2.*) as encounters,
                coalesce(sum(m2.sets_won::double precision/greatest(1,coalesce(m2.num_sets,0))), 0) as match_closeness,
                sum(m2.games_won+m2.games_against)/greatest(1, sum(m2.num_sets)) as games_per_set,
                count(distinct m2.tournament) as num_tournaments,
                sum(coalesce(t2.masters,0))/greatest(count(m2.*),1) as avg_tournament_rank,
                coalesce(sum(case when m2.player_victory then t2.masters else 0 end), 0) as atp_points,
                sum(case when t.masters <= t2.masters then 1.0 else 0.0 end) as better_encounters,
                sum(case when t.masters >= t2.masters then 1.0 else 0.0 end) as worse_encounters
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
                where m2.player_victory is not null
                group by m.player_id,m.opponent_id,m.start_date,m.tournament
            ) as temp order by player_id,start_date,tournament,random()            
        );
    '''.replace("{{N}}", str(i)).replace("{{MONTHS_START}}", str(i * 3)).replace("{{MONTHS_END}}", str(i * 3 + 3))
    print('Sql:', sql)
    cursor.execute(sql)
    conn.commit()
    print("Executed query...", i)
    cursor.close()


num_prior_quarters = 12
threads = []
for i in range(num_prior_quarters):
    thread = Thread(target=run, args=(i,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()


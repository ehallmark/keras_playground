import psycopg2 as pg

try:
    conn = pg.connect("dbname='ib_db' user='postgres' password='password'")
    cursor = conn.cursor()
except:
    print("I am unable to connect to the database.")
    exit(1)


def insert_stock_price(table_name, price, tick_type):
    cursor.execute('insert into '+table_name+' (price,tick_type,')



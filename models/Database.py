import psycopg2 as pg
from datetime import datetime
try:
    conn = pg.connect("dbname='ib_db' user='postgres' password='password'")
    cursor = conn.cursor()
except:
    print("I am unable to connect to the database.")
    exit(1)


def commit():
    conn.commit()


def close():
    conn.close()


def insert_stock_price(table_name, price, can_auto_execute, tick_type):
    if can_auto_execute > 0:
        can_auto_execute = "'t'"
    else:
        can_auto_execute = "'f'"

    cursor.execute('insert into ' + table_name +
                   ' (price,can_auto_execute,tick_type,created_at) values ('
                   + str(price)+', '+can_auto_execute+','+str(tick_type)+",'"+
                   datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"')")


def insert_stock_price_handler(table_name, data):
    insert_stock_price(table_name, data[2], data[1], data[3])


def insert_stock_size(table_name, size, tick_type):
    cursor.execute('insert into ' + table_name +
                   ' (size,tick_type,created_at) values ('
                   + str(size)+','+str(tick_type)+",'"+
                   datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"')")


def insert_stock_size_handler(table_name, data):
    insert_stock_size(table_name, data[2], data[1])


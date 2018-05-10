import psycopg2


try:
    conn = psycopg2.connect("dbname='ib_db' user='postgres' password='password'")
except:
    print("I am unable to connect to the database.")
    exit(1)


def insertStock()


\connect ib_db

drop table tick_price_aapl; -- drops the table
create table tick_price_aapl ( -- creates the table
    id serial primary key,
    tick_type integer,
    can_auto_execute boolean,
    price numeric(10,2),
    created_at timestamp
);

drop table tick_size_aapl; -- drops the table
create table tick_size_aapl ( -- creates the table
    id serial primary key,
    tick_type integer,
    size integer,
    created_at timestamp
);
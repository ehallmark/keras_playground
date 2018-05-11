\connect ib_db

create table tick_price_aapl (
    id serial primary key,
    tick_type integer,
    price numeric(10,2),
    created_at timestamp
);

create table tick_size_aapl (
    id serial primary key,
    tick_type integer,
    size integer,
    create_at timestamp
);
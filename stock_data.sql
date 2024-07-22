create database if not exists stock;

use stock;

create table test_dataset (
    id int auto_increment primary key,
    stock_name varchar(100) not null,
    stock_code varchar(100) not null,
    date date not null,
    time time not null,
    content text,
    vix float,
    SP_500 float,
    NY_Dow float,
    despair float,
    optimism float,
    concern float,
    excitement float,
    stability float,
    value float
);

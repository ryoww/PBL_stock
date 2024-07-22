create database if not exists stock;

use stock;

create table test_dataset (
    id int auto_increment primary key,
    stock_name varchar(100) not null,
    stock_code varchar(100) not null,
    date date not null,
    time time not null,
    headline text,
    content text,
    vix float,
    SP_500 float,
    NASDAQ float,
    NY_Dow float,
    headline_despair float,
    headline_optimism float,
    headline_concern float,
    headline_excitement float,
    headline_stability float,
    content_despair float,
    content_optimism float,
    content_concern float,
    content_excitement float,
    content_stability float,
    value float
);

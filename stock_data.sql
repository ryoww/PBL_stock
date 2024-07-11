create database if not exists stock;

use stock;

create table stock_data (
    id int auto_increment primary key,
    stock_name varchar(100) not null,
    stock_code varchar(100) not null,
    datetime datetime not null,
    content text,
    value float
);

create table ml_dataset (
    id int auto_increment primary key,
    stock_name varchar(100) not null,
    stock_code varchar(100) not null,
    datetime datetime not null,
    despair float not null,
    optimism float not null,
    concern float not null,
    excitement float not null,
    stability float not null
);

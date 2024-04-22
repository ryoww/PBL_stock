use actix_web::{middleware::Logger, web, App, HttpResponse, HttpServer, Responder};
use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use sqlx::MySqlPool;
use std::env;

static TABLES: [&str; 2] = ["users", "test"];
static COLUMNS: [&str; 4] = ["id", "name", "email", "company_id"];

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct User {
    id: i32,
    name: String,
    email: String,
    company_id: i32,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct GetColumn {
    table: String,
    column: String,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct ColumnData {
    id: i32,
    data: String,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct GetId {
    table: String,
    column: String,
    conditions: String,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct IdData {
    id: i32,
    data: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateUser {
    table: String,
    name: String,
    email: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Update {
    table: String,
    column: String,
    data: String,
    update_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Delete {
    table: String,
    id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetDatas {
    table: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateTable {
    table_name: String,
}

// ECHO
async fn echo(req_body: String) -> impl Responder {
    println!("Received : {}", req_body);
    HttpResponse::Ok().body(req_body)
}

// CREATE user
async fn create_user(pool: web::Data<MySqlPool>, user: web::Json<CreateUser>) -> impl Responder {
    println!("Creating user: {:?}", user);
    if !TABLES.contains(&user.table.as_str()) {
        return HttpResponse::BadRequest().body(format!("{} is not registered", user.table));
    }

    let columns = COLUMNS.join(", ");
    let query = format!("INSERT INTO {} ({}) VALUES (?, ?, ?)", user.table, columns);

    let result = sqlx::query(&query)
        .bind(&user.name)
        .bind(&user.email)
        .execute(&**pool)
        .await;

    match result {
        Ok(_) => {
            println!("User created successfully.");
            HttpResponse::Created().finish()
        }
        Err(e) => {
            println!("Failed to create user: {}", e);
            HttpResponse::InternalServerError().body("Failed to create user")
        }
    }
}

// GET table
async fn get_users_table(pool: web::Data<MySqlPool>, req: web::Json<GetDatas>) -> impl Responder {
    println!("Fetching users from table: {}", req.table);
    if !TABLES.contains(&req.table.as_str()) {
        return HttpResponse::BadRequest().body(format!("{} is not registered", req.table));
    }

    let columns = COLUMNS.join(", ");
    let query = format!("SELECT {} FROM {}", columns, req.table);
    println!("Executing query: {}", query);

    let result = sqlx::query_as::<_, User>(&query)
        .fetch_all(pool.get_ref())
        .await;

    match result {
        Ok(users) => {
            println!("Query successful, retrieved {}.", req.table);
            HttpResponse::Ok().json(users)
        }
        Err(e) => {
            println!("Database error: {}", e);
            HttpResponse::InternalServerError().body(format!("Database error: {}", e))
        }
    }
}


// CREATE table
async fn create_table(pool: web::Data<MySqlPool>, table_name: web::Json<CreateTable>) -> impl Responder {
    println!("Creating table: {}", table_name.table_name);
    if TABLES.contains(&table_name.table_name.as_str()) {
        return HttpResponse::BadRequest().body(format!("Table '{}' is registered", table_name.table_name));
    }

    let query = format!(
        "CREATE TABLE IF NOT EXISTS {} (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255))",
        table_name.table_name
    );

    let result = sqlx::query(&query)
    .execute(&**pool)
    .await;

    match result {
        Ok(_) => {
            println!("Table '{}' created successfully.", table_name.table_name);
            HttpResponse::Created().body(format!("Table '{}' created successfully", table_name.table_name))
        }
        Err(e) => {
            println!("Failed to create table '{}': {}", table_name.table_name, e);
            HttpResponse::InternalServerError().body(format!("Failed to create table '{}'", table_name.table_name))
        }
    }
}

// GET id
async fn get_users_id(pool: web::Data<MySqlPool>, req: web::Json<GetId>) -> impl Responder {
    if !TABLES.contains(&req.table.as_str()) {
        return HttpResponse::BadRequest().body(format!("{} is not registered", req.table));
    } else if !COLUMNS.contains(&req.column.as_str()) {
        return HttpResponse::BadRequest().body(format!("{} is not registered", req.column));
    }

    let query = format!(
        "SELECT id, {} AS data FROM {} WHERE {}",
        req.column, req.table, req.conditions
    );
    println!("Executing query: {}", query);

    let result = sqlx::query_as::<_, IdData>(&query)
        .fetch_all(pool.get_ref())
        .await;

    match result {
        Ok(iddata) => {
            println!("Query successful, retrieved {}.", req.table);
            HttpResponse::Ok().json(iddata)
        }
        Err(e) => {
            println!("Database error: {}", e);
            HttpResponse::InternalServerError().body(format!("Database error: {}", e))
        }
    }
}

// GET column
async fn get_users_clumn(pool: web::Data<MySqlPool>, req: web::Json<GetColumn>) -> impl Responder {
    if !TABLES.contains(&req.table.as_str()) {
        return HttpResponse::BadRequest().body(format!("{} is not registered", req.table));
    } else if !COLUMNS.contains(&req.column.as_str()) {
        return HttpResponse::BadRequest().body(format!("{} is not registered", req.column));
    }

    let query = format!("SELECT id, {} as data FROM {}", req.column, req.table);
    println!("Executing query: {}", query);

    let result = sqlx::query_as::<_, ColumnData>(&query)
        .fetch_all(pool.get_ref())
        .await;

    match result {
        Ok(data) => {
            println!("Query successful, retrieved {}.", req.table);
            HttpResponse::Ok().json(data)
        }
        Err(e) => {
            println!("Database error: {}", e);
            HttpResponse::InternalServerError().body(format!("Database error: {}", e))
        }
    }
}

// UPDATE safety
async fn update_user(pool: web::Data<MySqlPool>, update: web::Json<Update>) -> impl Responder {
    println!("Updating user: {}", update.update_id);
    let valid_columns = vec!["name", "email", "company_id"];

    if !TABLES.contains(&update.table.as_str()) {
        return HttpResponse::BadRequest().body(format!("{} is not registered", update.table));
    } else if !valid_columns.contains(&update.column.as_str()) {
        return HttpResponse::BadRequest().body("Invalid column name");
    }

    let query = format!(
        "UPDATE {} SET {} = ? WHERE id = ?",
        update.table, update.column
    );

    let result = sqlx::query(&query)
        .bind(&update.data)
        .bind(&update.update_id)
        .execute(&**pool)
        .await;

    match result {
        Ok(_) => {
            println!("Update successful.");
            HttpResponse::Created().finish()
        }
        Err(e) => {
            println!("Failed to update user: {}", e);
            HttpResponse::InternalServerError().body("Failed to update user")
        }
    }
}

// DELETE
async fn delete_user(pool: web::Data<MySqlPool>, req: web::Json<Delete>) -> impl Responder {
    println!("Deleting user with ID: {}", req.id);
    if !TABLES.contains(&req.table.as_str()) {
        return HttpResponse::BadRequest().body(format!("{} is not registered", req.table));
    }

    let query = format!("DELETE FROM {} WHERE id = {}", req.table, req.id);
    let result = sqlx::query(&query).execute(&**pool).await;

    match result {
        Ok(_) => {
            println!("User deleted successfully.");
            HttpResponse::Ok().body("User deleted successfully")
        }
        Err(e) => {
            println!("Failed to delete user: {}", e);
            HttpResponse::InternalServerError().body("Failed to delete user")
        }
    }
}

async fn create_db_pool() -> MySqlPool {
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    MySqlPool::connect(&database_url)
        .await
        .expect("Failed to create pool")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    env_logger::init();

    let pool = create_db_pool().await;

    HttpServer::new(move || {
        App::new()
            .route("/", web::post().to(echo))
            .app_data(web::Data::new(pool.clone()))
            .wrap(Logger::default())
            .route("/users", web::post().to(create_user))
            .route("/users", web::get().to(get_users_table))
            .route("/column", web::get().to(get_users_clumn))
            .route("/update", web::post().to(update_user))
            .route("/delete", web::delete().to(delete_user))
            .route("/getid", web::get().to(get_users_id))
            .route("/create", web::post().to(create_table))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

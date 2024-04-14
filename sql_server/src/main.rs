use actix_web::{middleware::Logger, web, App, HttpResponse, HttpServer, Responder};
use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, MySqlPool};
use std::{env, result};

static TABLES: [&str; 2] = ["users", "table"];
static COLUMNS: [&str; 3] = ["id", "name", "email"];

#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    id: i32,
    name: String,
    email: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateUser {
    table: String,
    name: String,
    email: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Update {
    column: String,
    data: String,
    update_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Delete {
    id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetDatas {
    table: String,
    columns: Vec<String>,
}

async fn is_columns(columns: &Vec<String>) -> bool {
    for e in columns {
        if !COLUMNS.contains(&e.as_str()) {
            return false;
        }
    }

    return true;
}

// ECHO
async fn echo(req_body: String) -> impl Responder {
    println!("Received : {}", req_body);
    HttpResponse::Ok().body(req_body)
}

// CREATE user
async fn create_user(pool: web::Data<MySqlPool>, user: web::Json<CreateUser>) -> impl Responder {
    if !TABLES.contains(&user.table.as_str()) {
        return HttpResponse::BadRequest().body(format!("{} is no registered", user.table));
    }

    let query = format!(
        "INSERT INTO {} (name, email) VALUES ({}, {})",
        user.table, user.name, user.email
    );

    let _ = sqlx::query(&query)
        .execute(&**pool)
        .await
        .expect("Failed to execute query");

    HttpResponse::Created().finish()
}

// GET users
async fn get_users_table(pool: web::Data<MySqlPool>, req: web::Json<GetDatas>) -> impl Responder {
    if !TABLES.contains(&req.table.as_str()) {
        return HttpResponse::BadRequest().body(format!("{} is not registered", req.table));
    } else if !is_columns(&req.columns).await {
        return HttpResponse::BadRequest()
            .body(format!("{} is no registered", req.columns.join(", ")));
    }

    let columns = req.columns.join(", ");

    let query = format!("SELECT {} FROM {}", columns, req.table);

    let result = sqlx::query_as::<_, User>(&query)
        .fetch_all(pool.get_ref())
        .await;

    match result {
        Ok(users) => HttpResponse::Ok().json(users),
    }
}

// UPDATE safety
async fn update_user(pool: web::Data<MySqlPool>, update: web::Json<Update>) -> impl Responder {
    let valid_columns = vec!["name", "email"];
    if !valid_columns.contains(&update.column.as_str()) {
        return HttpResponse::BadRequest().body("Invalid column name");
    }

    let query = format!("UPDATE users SET {} = ? WHERE id = ?", update.column);
    let _ = sqlx::query(&query)
        .bind(&update.data)
        .bind(&update.update_id)
        .execute(&**pool)
        .await
        .expect("Failed to execute query");

    HttpResponse::Created().finish()
}

// DELETE
async fn delete_user(pool: web::Data<MySqlPool>, req: web::Json<Delete>) -> impl Responder {
    let result = sqlx::query!("DELETE FROM test WHERE id = ?", req.id)
        .execute(&**pool)
        .await;

    match result {
        Ok(_) => HttpResponse::Ok().body("User deleted successfully"),
        Err(e) => {
            println!("Failed to Delete user {}", e);
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
            .route("/update", web::post().to(update_user))
            .route("/delete", web::post().to(delete_user))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

use actix_web::{middleware::Logger, web, App, HttpResponse, HttpServer, Responder};
use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use sqlx::MySqlPool;
use std::env;

#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    table: String,
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
    table: String,
    column: String,
    data: String,
    update_id: String
}

async fn echo(req_body: String) -> impl Responder {
    println!("Received : {}", req_body);
    HttpResponse::Ok().body(req_body)
}

async fn create_user(pool: web::Data<MySqlPool>, user: web::Json<CreateUser>) -> impl Responder {
    let _ = sqlx::query!(
        "INSERT INTO test (name, email) VALUES (?, ?)",
        &user.table,
        &user.name,
        &user.email,
    )
    .execute(&**pool)
    .await
    .expect("Failed to execute query");

    HttpResponse::Created().finish()
}

async fn get_users_table(pool: web::Data<MySqlPool>) -> impl Responder {
    let result = sqlx::query_as!(User, "SELECT id, name, email FROM test")
        .fetch_all(&**pool)
        .await
        .expect("Failed to execute query");

    HttpResponse::Ok().json(result)
}

async fn update_user(pool: web::Data<MySqlPool>, update: web::Json<Update) -> impl Responder {
    let _ = sqlx::query!(
        "UPDATE ? SET ? = ? WHERE id = ?",
        &update.table,
        &update.column,
        &update.data,
        &update.update_id
    )
    .execute(&**pool)
    .await
    .expect("Failed to execute query");

    HttpResponse::Created().finish()
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
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

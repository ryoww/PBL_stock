use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use dotenv::dotenv;
use sqlx::MySqlPool;
use std::env;

async fn echo(req_body: String) -> impl Responder {
    println!("Received : {}", req_body);
    HttpResponse::Ok().body(req_body)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().route("/", web::post().to(echo)))
        .bind(("127.0.0.1", 8080))?
        .run()
        .await
}

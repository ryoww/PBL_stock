use actix_web::{middleware::Logger, web, App, HttpResponse, HttpServer, Responder};
use chrono::{NaiveDateTime, ParseError};
use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use sqlx::MySqlPool;
use log::{info, error};
use std::env;

static COLUMNS: [&str; 4] = ["value", "vix", "SP_500", "NY_Dow"];

#[derive(Debug, Serialize, Deserialize)]
pub struct PostRowData {
    stock_name: String,
    stock_code: String,
    datetime: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PostMLData {
    despair: f64,
    optimism: f64,
    concern: f64,
    excitement: f64,
    stability: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateColumnData {
    date: String,
    column_name: String,
    update_value: String,
    stock_code: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct GetRowData {
    id: i32,
    stock_name: String,
    stock_code: String,
    date: String,
    time: String,
    content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct MLData {
    date: String,
    time: String,
    despair: Option<f64>,
    optimism: Option<f64>,
    concern: Option<f64>,
    excitement: Option<f64>,
    stability: Option<f64>,
    value: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct MaxId {
    max_id: Option<i64>,
}

// エラー応答の共通化
fn create_error_response(message: &str) -> HttpResponse {
    error!("{}", message);
    HttpResponse::InternalServerError().body(message.to_string())
}

// 日時文字列を日付と時間に分割する
fn parse_and_split_datetime(datetime_str: &str) -> Result<(String, String), ParseError> {
    let datetime = NaiveDateTime::parse_from_str(datetime_str, "%Y-%m-%d %H:%M:%S")?;
    Ok((datetime.date().to_string(), datetime.time().to_string()))
}

// DBに生データをPOST
async fn add_row_data(pool: web::Data<MySqlPool>, rowdata: web::Json<PostRowData>) -> impl Responder {
    info!("Adding data: {:?}", rowdata);

    let (date, time) = match parse_and_split_datetime(&rowdata.datetime) {
        Ok(dt) => dt,
        Err(_) => return HttpResponse::BadRequest().body("Invalid datetime format"),
    };

    let query = "INSERT INTO stock_dataset (stock_name, stock_code, date, time, content) VALUES (?, ?, ?, ?, ?)";

    match sqlx::query(query)
        .bind(&rowdata.stock_name)
        .bind(&rowdata.stock_code)
        .bind(date)
        .bind(time)
        .bind(&rowdata.content)
        .execute(&**pool)
        .await {
        Ok(_) => HttpResponse::Created().finish(),
        Err(e) => create_error_response(&format!("Failed to add data: {}", e)),
    }
}

// DBから生データをGET
async fn get_row_data(pool: web::Data<MySqlPool>, path: web::Path<i32>) -> impl Responder {
    let id = path.into_inner();
    info!("Fetching data for ID: {}", id);

    let query = "SELECT stock_name, stock_code, date, time, content FROM stock_dataset WHERE id = ?";

    match sqlx::query_as::<_, GetRowData>(query)
        .bind(id)
        .fetch_one(pool.get_ref())
        .await {
        Ok(data) => HttpResponse::Ok().json(data),
        Err(e) => create_error_response(&format!("Failed to fetch data: {}", e)),
    }
}

// DBのidのMAXを返す
async fn get_len(pool: web::Data<MySqlPool>) -> impl Responder {
    info!("Getting the max id of the table");

    let query = "SELECT MAX(id) as max_id FROM stock_dataset";

    match sqlx::query_as::<_, MaxId>(query)
        .fetch_one(pool.get_ref())
        .await {
            Ok(max_id) => HttpResponse::Ok().json(max_id),
            Err(e) => create_error_response(&format!("Failed to get max id: {}", e)),
        }
}

// DBにMLパラメータをPOST
async fn post_ml_data(pool: web::Data<MySqlPool>, path: web::Path<i32>, ml_data: web::Json<PostMLData>) -> impl Responder {
    let id = path.into_inner();
    info!("Posting ML data for ID: {}", id);

    let query = "UPDATE stock_dataset SET despair = ?, optimism = ?, concern = ?, excitement = ?, stability = ? WHERE id = ?";

    match sqlx::query(query)
        .bind(ml_data.despair)
        .bind(ml_data.optimism)
        .bind(ml_data.concern)
        .bind(ml_data.excitement)
        .bind(ml_data.stability)
        .bind(id)
        .execute(&**pool)
        .await {
        Ok(_) => HttpResponse::Ok().body("ML data updated successfully"),
        Err(e) => create_error_response(&format!("Failed to update ML data: {}", e)),
    }
}

// DBからML用データセットをGET
async fn get_ml_data(pool: web::Data<MySqlPool>, path: web::Path<String>) -> impl Responder {
    let stock_code = path.into_inner();
    info!("Fetching ML data for stock_code: {}", stock_code);

    let query = "SELECT date, time, despair, optimism, concern, excitement, stability, value, vix, SP_500, NY_Dow FROM stock_dataset WHERE stock_code = ?";

    match sqlx::query_as::<_, MLData>(query)
        .bind(stock_code)
        .fetch_all(pool.get_ref())
        .await {
        Ok(data) => HttpResponse::Ok().json(data),
        Err(e) => create_error_response(&format!("Failed to fetch ML data: {}", e)),
    }
}

// DBのcolumnをUPDATE
async fn update_column(pool: web::Data<MySqlPool>, update_data: web::Json<UpdateColumnData>) -> impl Responder {
    info!("Updating column: {} on date: {}", update_data.column_name, update_data.date);

    // ベースのクエリ
    let mut query = format!("UPDATE stock_dataset SET {} = ? WHERE date = ?", update_data.column_name);

    // stock_codeが指定されている場合は条件に追加
    if let Some(_stock_code) = &update_data.stock_code {
        if !COLUMNS.contains(&update_data.column_name.as_str()) {
            return HttpResponse::BadRequest().body(format!("{} is not registered", update_data.column_name));
        }

        query.push_str(" AND stock_code = ?");
    }

    let mut sql_query = sqlx::query(&query)
        .bind(&update_data.update_value)
        .bind(&update_data.date);

    if let Some(stock_code) = &update_data.stock_code {
        sql_query = sql_query.bind(stock_code);
    }

    let result = sql_query.execute(&**pool).await;

    match result {
        Ok(result) => {
            if result.rows_affected() == 0 {
                HttpResponse::NotFound().body("No record found for the given date and stock_code")
            } else {
                HttpResponse::Ok().body("Column updated successfully")
            }
        }
        Err(e) => create_error_response(&format!("Failed to update column: {}", e)),
    }
}

// DBプールの作成
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
            .app_data(web::Data::new(pool.clone()))
            .wrap(Logger::default())
            .route("/", web::post().to(|body: String| async move { HttpResponse::Ok().body(body) }))
            .route("/row_data", web::post().to(add_row_data))
            .route("/row_data/{id}", web::get().to(get_row_data))
            .route("/get_len", web::get().to(get_len))
            .route("/ml_data/{id}", web::post().to(post_ml_data))
            .route("/ml_data/{stock_code}", web::get().to(get_ml_data))
            .route("/update_column", web::post().to(update_column))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

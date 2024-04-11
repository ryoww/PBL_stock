use sqlx::MySqlPool;
use dotenv::dotenv;
use std::env;

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    // .envファイルから環境変数をロード
    dotenv().ok();
    
    // DATABASE_URL環境変数からデータベース接続情報を取得
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    println!("{}", database_url);
    
    // データベースプールを作成（接続テスト）
    let pool = MySqlPool::connect(&database_url).await?;
    
    println!("データベースへの接続に成功しました。");

    // 接続テスト用のクエリを実行
    let row: (i64,) = sqlx::query_as("SELECT 1")
        .fetch_one(&pool).await?;
    
    println!("テストクエリの実行結果: {}", row.0);

    Ok(())
}

from flask import Flask, request, jsonify, make_response
from flask_marshmallow import Marshmallow
from mysql.connector import pooling, Error
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()
DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME")
}

# Create connection pool
connection_pool = pooling.MySQLConnectionPool(pool_name="mypool",
                                              pool_size=10,
                                              **DB_CONFIG)

# Flask setup
app = Flask(__name__)
ma = Marshmallow(app)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Marshmallow Schemas
class PostRowDataSchema(ma.Schema):
    class Meta:
        fields = ("stock_name", "stock_code", "datetime", "headline", "content")

class PostMLDataSchema(ma.Schema):
    class Meta:
        fields = (
            "headline_despair", "headline_optimism", "headline_concern", "headline_excitement", "headline_stability",
            "content_despair", "content_optimism", "content_concern", "content_excitement", "content_stability"
        )

class UpdateColumnDataSchema(ma.Schema):
    class Meta:
        fields = ("date", "column_name", "update_value", "stock_code")

class SpotUpdateSchema(ma.Schema):
    class Meta:
        fields = ("id", "column_name", "update_value")

class SpotGetSchema(ma.Schema):
    class Meta:
        fields = ("id", "column_name")

post_row_data_schema = PostRowDataSchema()
post_ml_data_schema = PostMLDataSchema()
update_column_data_schema = UpdateColumnDataSchema()
spot_update_schema = SpotUpdateSchema()
spot_get_schema = SpotGetSchema()

ALLOWED_COLUMNS = ['vix', 'SP_500', 'NY_Dow', 'NASDAQ']

# Helper function to get connection from the pool
def get_db_connection():
    try:
        connection = connection_pool.get_connection()
        if connection.is_connected():
            return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        raise

# Function to convert timedelta to string
def timedelta_to_str(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# Routes
@app.route('/', methods=['GET'])
def hellow():
    return make_response('Hello Server\n', 200)

@app.route("/row_data", methods=["POST"])
def add_row_data():
    rowdata = request.get_json()
    errors = post_row_data_schema.validate(rowdata)
    if errors:
        return make_response(jsonify(errors), 400)
    data = post_row_data_schema.load(rowdata)

    logger.info(f"Adding data: {data}")

    try:
        datetime_obj = datetime.strptime(data['datetime'], "%Y-%m-%d %H:%M")
        date = datetime_obj.date()
        time = datetime_obj.time()
    except ValueError:
        return make_response("Invalid datetime format", 400)

    query = """
    INSERT INTO stock_dataset (stock_name, stock_code, date, time, headline, content)
    VALUES (%s, %s, %s, %s, %s, %s)
    """

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(query, (data['stock_name'], data['stock_code'], date, time, data['headline'], data['content']))
        connection.commit()
        print("Executed query:", cursor.statement)
        return make_response("Data added successfully", 201)
    except Error as e:
        logger.error(f"Error adding data: {e}")
        print("Error message:", e)
        return make_response("Failed to add data", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/row_data/<int:id>", methods=["GET"])
def get_row_data(id):
    logger.info(f"Fetching data for ID: {id}")

    query = """
    SELECT id, stock_name, stock_code, date, time, headline, content
    FROM stock_dataset
    WHERE id = %s
    """

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query, (id,))
        print("Executed query:", cursor.statement)
        row = cursor.fetchone()
        print("Fetched row:", row)
        if row:
            row['date'] = row['date'].isoformat()
            row['time'] = timedelta_to_str(row['time']) if isinstance(row['time'], timedelta) else row['time'].isoformat()
            return jsonify(row)
        else:
            return make_response("Data not found", 404)
    except Error as e:
        logger.error(f"Error fetching data: {e}")
        print("Error message:", e)
        return make_response("Failed to fetch data", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/get_len", methods=["GET"])
def get_len():
    logger.info("Getting the max id of the table")

    query = "SELECT MAX(id) AS max_id FROM stock_dataset"

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query)
        print("Executed query:", cursor.statement)
        result = cursor.fetchone()
        print("Max ID:", result)
        return jsonify(result)
    except Error as e:
        logger.error(f"Error fetching max id: {e}")
        print("Error message:", e)
        return make_response("Failed to get max id", 500)
    finally: 
        cursor.close()
        connection.close()

@app.route("/get_min_null_id", methods=["GET"])
def get_min_null_id():
    logger.info("Getting the min null id of the table")
    
    query = "SELECT MIN(id) AS min_null_id FROM stock_dataset WHERE headline_despair IS NULL"
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute(query)
        print("Executed query:", cursor.statement)
        result = cursor.fetchone()
        print("Max ID:", result)
        return jsonify(result)
    except Error as e:
        logger.error(f"Error fetching max id: {e}")
        print("Error message:", e)
        return make_response("Failed to get max id", 500)
    finally: 
        cursor.close()
        connection.close()

@app.route("/ml_data/<int:id>", methods=["POST"])
def post_ml_data(id):
    ml_data = request.get_json()
    errors = post_ml_data_schema.validate(ml_data)
    if errors:
        return make_response(jsonify(errors), 400)
    data = post_ml_data_schema.load(ml_data)

    logger.info(f"Posting ML data for ID: {id}")

    query = """
    UPDATE stock_dataset
    SET headline_despair = %s, headline_optimism = %s, headline_concern = %s, headline_excitement = %s, headline_stability = %s, content_despair = %s, content_optimism = %s, content_concern = %s, content_excitement = %s, content_stability = %s
    WHERE id = %s
    """

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(query, (
            data['headline_despair'], data['headline_optimism'], data['headline_concern'], data['headline_excitement'], data['headline_stability'],
            data['content_despair'], data['content_optimism'], data['content_concern'], data['content_excitement'], data['content_stability'], id
        ))
        connection.commit()
        print("Executed query:", cursor.statement)
        if cursor.rowcount == 0:
            return make_response("Data not found", 404)
        return make_response("ML data updated successfully", 200)
    except Error as e:
        logger.error(f"Error updating ML data: {e}")
        print("Error message:", e)
        return make_response("Failed to update ML data", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/ml_data/<string:stock_code>", methods=["GET"])
def get_ml_data(stock_code):
    logger.info(f"Fetching ML data for stock_code: {stock_code}")

    query = """
    SELECT date, time, headline_despair, headline_optimism, headline_concern, headline_excitement, headline_stability, content_despair, content_optimism, content_concern, content_excitement, content_stability, vix, SP_500, NY_Dow, value
    FROM stock_dataset
    WHERE stock_code = %s
    """

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query, (stock_code,))
        print("Executed query:", cursor.statement)
        rows = cursor.fetchall()
        print("Fetched rows:", rows)
        for row in rows:
            row['date'] = row['date'].isoformat()
            row['time'] = timedelta_to_str(row['time']) if isinstance(row['time'], timedelta) else row['time'].isoformat()
        return jsonify(rows)
    except Error as e:
        logger.error(f"Error fetching ML data: {e}")
        print("Error message:", e)
        return make_response("Failed to fetch ML data", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/update_column", methods=["POST"])
def update_column():
    update_data = request.get_json()
    errors = update_column_data_schema.validate(update_data)
    if errors:
        logger.error(f"Validation errors: {errors}")
        return make_response(jsonify(errors), 400)
    data = update_column_data_schema.load(update_data)

    logger.info(f"Updating column: {data['column_name']} on date: {data['date']} with stock_code: {data.get('stock_code')}")

    query = f"""
    UPDATE stock_dataset
    SET {data['column_name']} = %s
    WHERE date = %s
    """

    if data.get('stock_code'):
        query += f""" AND stock_code = %s"""

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        if data.get('stock_code'):
            cursor.execute(query, (data['update_value'], data['date'], data['stock_code']))
        else:
            cursor.execute(query, (data['update_value'], data['date']))
        connection.commit()
        logger.info(f"Executed query: {cursor.statement}")
        print("Executed query:", cursor.statement)
        print("Affected rows:", cursor.rowcount)
        if cursor.rowcount == 0:
            logger.warning("No record found for the given date and stock_code")
            return make_response("No record found for the given date and stock_code", 404)
        return make_response("Column updated successfully", 200)
    except Error as e:
        logger.error(f"Error updating column: {e}")
        print("Error message:", e)
        return make_response("Failed to update column", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/getdays/<string:stock_code>", methods=["GET"])
def get_days(stock_code):
    # nullパラメータをクエリパラメータから取得
    null_param = request.args.get('null', 'false').lower() == 'true'

    logger.info(f"Fetching datetime for stock_code: {stock_code} with null condition: {null_param}")

    # 基本のクエリ
    query = """
    SELECT date, time
    FROM stock_dataset
    WHERE stock_code = %s
    """

    # null_paramがTrueの場合、AND value IS NULLをクエリに追加
    if null_param:
        query += " AND value IS NULL"

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query, (stock_code,))
        print("Executed query:", cursor.statement)
        rows = cursor.fetchall()
        for row in rows:
            row['date'] = row['date'].isoformat()
            row['time'] = timedelta_to_str(row['time']) if isinstance(row['time'], timedelta) else row['time'].isoformat()
        return jsonify(rows)
    except Error as e:
        logger.error(f"Error fetching datetime: {e}")
        print("Error message:", e)
        return make_response("Failed to fetch datetime", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/spot_update", methods=["POST"])
def spot_update():
    update_data = request.get_json()
    errors = spot_update_schema.validate(update_data)
    if errors:
        logger.error(f"Validation errors: {errors}")
        return make_response(jsonify(errors), 400)
    data = spot_update_schema.load(update_data)

    logger.info(f"Updating column: {data['column_name']} for ID: {data['id']}")

    query = f"""
    UPDATE stock_dataset
    SET {data['column_name']} = %s
    WHERE id = %s
    """

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(query, (data['update_value'], data['id']))
        connection.commit()
        logger.info(f"Executed query: {cursor.statement}")
        print("Executed query:", cursor.statement)
        print("Affected rows:", cursor.rowcount)
        if cursor.rowcount == 0:
            logger.warning("No record found for the given ID")
            return make_response("No record found for the given ID", 404)
        return make_response("Column updated successfully", 200)
    except Error as e:
        logger.error(f"Error updating column: {e}")
        print("Error message:", e)
        return make_response("Failed to update column", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/spot_get", methods=["POST"])
def spot_get():
    get_data = request.get_json()
    errors = spot_get_schema.validate(get_data)
    if errors:
        logger.error(f"Validation errors: {errors}")
        return make_response(jsonify(errors), 400)
    data = spot_get_schema.load(get_data)

    logger.info(f"Fetching column: {data['column_name']} for ID: {data['id']}")

    query = f"""
    SELECT {data['column_name']}
    FROM stock_dataset
    WHERE id = %s
    """

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query, (data['id'],))
        print("Executed query:", cursor.statement)
        row = cursor.fetchone()
        print("Fetched row:", row)
        if row:
            return jsonify(row)
        else:
            return make_response("Data not found", 404)
    except Error as e:
        logger.error(f"Error fetching data: {e}")
        print("Error message:", e)
        return make_response("Failed to fetch data", 500)
    finally:
        cursor.close()
        connection.close()

# 新しいエンドポイント: 特定のカラムを取得する
@app.route("/get_column/<string:column_name>", methods=["GET"])
def get_column_data(column_name):
    logger.info(f"Fetching column: {column_name} for all records")

    query = f"SELECT {column_name} FROM stock_dataset"

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query)
        print("Executed query:", cursor.statement)
        rows = cursor.fetchall()
        print("Fetched rows:", rows)
        return jsonify(rows)
    except Error as e:
        logger.error(f"Error fetching column data: {e}")
        print("Error message:", e)
        return make_response("Failed to fetch column data", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/null_values/<string:stock_code>", methods=["GET"])
def get_null_values(stock_code):
    logger.info(f"Fetching null value IDs for stock_code: {stock_code}")

    query = """
    SELECT id, date
    FROM stock_dataset
    WHERE stock_code = %s AND value IS NULL
    """

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query, (stock_code,))
        print("Executed query:", cursor.statement)
        rows = cursor.fetchall()
        print("Fetched rows:", rows)
        
        # 日付をキーとしてIDのリストを値とする辞書を作成
        result = {}
        for row in rows:
            date_str = row['date'].isoformat()
            if date_str not in result:
                result[date_str] = []
            result[date_str].append(row['id'])

        return jsonify(result)
    except Error as e:
        logger.error(f"Error fetching null values: {e}")
        print("Error message:", e)
        return make_response("Failed to fetch null values", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/null_index/<string:index_code>", methods=["GET"])
def get_null_index(index_code):
    if index_code not in ALLOWED_COLUMNS:
        return make_response("Invalid column name", 400)
    
    logger.info(f"Fetching null value IDs for stock_code: {index_code}")

    query = f"""
    SELECT id, date
    FROM stock_dataset
    WHERE {index_code} IS NULL
    """

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query)  # パラメータなしで実行
        print("Executed query:", cursor.statement)
        rows = cursor.fetchall()
        print("Fetched rows:", rows)
        
        # 日付をキーとしてIDのリストを値とする辞書を作成
        result = {}
        for row in rows:
            date_str = row['date'].isoformat()
            if date_str not in result:
                result[date_str] = []
            result[date_str].append(row['id'])

        return jsonify(result)
    except Error as e:
        logger.error(f"Error fetching null values: {e}")
        print("Error message:", e)
        return make_response("Failed to fetch null values", 500)
    finally:
        cursor.close()
        connection.close()



# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8999)

